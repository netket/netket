# Copyright 2020 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from functools import singledispatch
from functools import singledispatch
from typing import Callable, Optional, Sequence, Union

import numpy as np

import flax
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from tqdm.std import tqdm
from netket.driver.abstract_variational_driver import _to_iterable

import netket as nk
from netket.driver import AbstractVariationalDriver
from netket.driver.vmc_common import info
from netket.logging.json_log import JsonLog
from netket.operator import AbstractOperator
from netket.optimizer import LinearOperator
from netket.optimizer.qgt import QGTAuto
from netket.utils import mpi
from netket.utils.types import Array, PyTree
from netket.vqs import VariationalState, MCState, MCMixedState

from netket.experimental.dynamics import RungeKuttaIntegrator, RKIntegratorConfig


@singledispatch
def dwdt(state, driver, t, w, *, stage=None):
    raise NotImplementedError(f"dwdt not implemented for {type(state)}")


@dwdt.register
def dwdt_mcstate(state: MCState, driver, t, w, *, stage: int = None):
    state.parameters = w
    state.reset()

    driver._loss_stats, driver._loss_grad = state.expect_and_grad(
        driver.generator(t),
        use_covariance=True,
    )
    driver._loss_grad = tree_map(
        lambda x: driver._loss_grad_factor * x, driver._loss_grad
    )

    qgt = driver.qgt(driver.state)
    if stage == 0:  # TODO: This does not work with FSAL.
        driver._last_qgt = qgt

    initial_dw = None if driver.linear_solver_restart else driver._dw
    driver._dw, _ = qgt.solve(driver.linear_solver, driver._loss_grad, x0=initial_dw)
    return driver._dw


class TimeDependentVMC(AbstractVariationalDriver):
    """
    Variational Time evolution using the time-dependent Variational Monte Carlo (t-VMC).
    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        integrator_config: RKIntegratorConfig,
        *,
        t0: float = 0.0,
        propagation_type="real",
        qgt: LinearOperator = None,
        linear_solver=None,
        linear_solver_restart: bool = False,
        error_norm: str = "euclidean",
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (Hamiltonian for pure states,
                Lindbladian for density operators).
            variational_state: The variational state.
            integrator_config: Configuration of the algorithm used for solving the ODE.
            t0: Initial time.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schödinger equation (SE), "imag" for the imaginary-time SE.
            qgt: The QGT specification.
            linear_solver: The solver for solving the linear system determining the time evolution.
            linear_solver_restart: If False (default), the last solution of the linear system
                is used as initial value in subsequent steps.
            error_norm: Norm function used to calculate the error with adaptive integrators.
                Can be either "euclidean" for the standard L2 vector norm :math:`x^\dagger \cdot x`
                or "qgt", in which case the scalar product induced by the QGT :math:`S` is used
                to compute the norm :math:`\Vert x \Vert^2_S = x^\dagger \cdot S \cdot x` as suggested
                in PRL 125, 100503 (2020).
        """
        self._t0 = t0

        if linear_solver is None:
            linear_solver = nk.optimizer.solver.svd
        if qgt is None:
            qgt = QGTAuto(solver=linear_solver)

        super().__init__(
            variational_state, optimizer=None, minimized_quantity_name="Generator"
        )

        if isinstance(operator, AbstractOperator):
            operator = operator.collect()
            self._generator = lambda _: operator
        else:
            self._generator = operator

        self.propagation_type = propagation_type
        if propagation_type == "real":
            self._loss_grad_factor = -1.0j
        elif propagation_type == "imag":
            self._loss_grad_factor = -1.0
        else:
            raise ValueError("propagation_type must be one of 'real', 'imag'")

        self.qgt = qgt
        self.linear_solver = linear_solver
        self.linear_solver_restart = linear_solver_restart

        self._w = self.state.parameters
        self._dw = None  # type: PyTree
        self._last_qgt = None

        self.integrator_config = integrator_config
        if error_norm == "euclidean":
            norm = None
        elif error_norm == "qgt":
            norm = self._qgt_norm
        else:
            raise ValueError("error_norm must be one of 'euclidean', 'qgt'")
        self._integrator = integrator_config(self._odefun, t0, self._w, norm=norm)
        self._stop_count = 0

    @property
    def generator(self) -> AbstractOperator:
        """
        The generator of the dynamics.
        """
        return self._generator

    @property
    def integrator(self):
        """
        The underlying integrator which computes the time steps.
        """
        return self._integrator

    def advance(self, T: float):
        """
        Advance the time propagation by `T` to `self.t + T`.

           Args:
               T: Length of the integration interval.
        """
        for _ in self.iter(T):
            pass

    def iter(self, T: float, *, tstops: Optional[Sequence[float]] = None):
        """
        Returns a generator which advances the time evolution in
        steps of `step` for a total of `n_iter` times.

        Args:
            T: Length of the integration interval.
            tstops: A sequence of stopping times, each within the intervall :code:`[self.t0, self.t0 + T]`,
                at which this method will stop and yield. By default, a stop is performed
                after each time step (at potentially varying step size if an adaptive
                integrator is used).
        Yields:
            The current step count.
        """
        t_end = self.t + T
        if tstops is not None and (
            np.any(np.less(tstops, self.t)) or np.any(np.greater(tstops, t_end))
        ):
            raise ValueError(f"All tstops must be in range [t, t + T]=[{self.t}, {T}]")

        if tstops is not None and len(tstops) > 0:
            tstops = np.sort(tstops)
            always_stop = False
        else:
            tstops = []
            always_stop = True

        while self.t < t_end:
            if always_stop or (
                len(tstops) > 0
                and (np.isclose(self.t, tstops[0]) or self.t > tstops[0])
            ):
                self._stop_count += 1
                yield self.t
                tstops = tstops[1:]

            step_accepted = False
            while not step_accepted:
                if not always_stop and len(tstops) > 0:
                    max_dt = tstops[0] - self.t
                else:
                    max_dt = None
                step_accepted = self._integrator.step(max_dt=max_dt)
            self._step_count += 1

        # Yield one last time if the remaining tstop is at t_end
        if (always_stop and np.isclose(self.t, t_end)) or (
            len(tstops) > 0 and np.isclose(tstops[0], t_end)
        ):
            yield self.t

    def run(
        self,
        T,
        out=None,
        obs=None,
        *,
        tstops=None,
        show_progress=True,
        callback=None,
    ):
        """
        Executes the Monte Carlo Variational optimization, updating the weights of the network
        stored in this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`. If no logger is specified, creates a json file at `out`,
        overwriting files with the same prefix.

        By default uses :ref:`netket.logging.JsonLog`. To know about the output format
        check it's documentation. The logger object is also returned at the end of this function
        so that you can inspect the results without reading the json output.

        Args:
            T: The integration time period.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing the observables that should be computed.
            tstops: A sequence of stopping times, each within the intervall :code:`[self.t0, self.t0 + T]`,
                at which the driver will stop and perform estimation of observables, logging, and excecute
                the callback function. By default, a stop is performed after each time step (at potentially
                varying step size if an adaptive integrator is used).
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to be executed at each
                stoping time.
        """
        if obs is None:
            obs = {}

        if callback is None:
            callback = lambda *_args, **_kwargs: True

        # Log only non-root nodes
        if self._mynode == 0:
            # if out is a path, create an overwriting Json Log for output
            if isinstance(out, str):
                loggers = (JsonLog(out, "w"),)
            else:
                loggers = _to_iterable(out)
        else:
            loggers = tuple()
            show_progress = False

        callbacks = _to_iterable(callback)
        callback_stop = False

        with tqdm(total=self.t + T, disable=not show_progress) as pbar:
            old_step = self.step_value
            first_step = True

            for step in self.iter(T, tstops=tstops):
                log_data = self.estimate(obs)

                # if the cost-function is defined then report it in the progress bar
                if self._loss_stats is not None:
                    pbar.set_postfix_str(self._loss_name + "=" + str(self._loss_stats))
                    log_data[self._loss_name] = self._loss_stats

                # Execute callbacks before loggers because they can append to log_data
                for callback in callbacks:
                    if not callback(step, log_data, self):
                        callback_stop = True

                for logger in loggers:
                    logger(self.step_value, log_data, self.state)

                if len(callbacks) > 0:
                    if mpi.mpi_any(callback_stop):
                        break

                # Reset the timing of tqdm after the first step, to ignore compilation time
                if first_step:
                    first_step = False
                    pbar.unpause()

                # Update the progress bar
                pbar.update(np.asarray(self.step_value - old_step))
                old_step = self.step_value

            # Final update so that it shows up filled.
            pbar.update(np.asarray(self.step_value - old_step))

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        return loggers

    def _log_additional_data(self, obs, step):
        obs["t"] = self.t

    @property
    def _default_step_size(self):
        # Essentially means
        return None

    @property
    def step_value(self):
        return self.t

    @property
    def dt(self):
        return self._integrator.dt

    @property
    def t(self):
        return self._integrator.t

    @t.setter
    def t(self, t):
        self._integrator.t = jnp.array(t, dtype=self._integrator.t)

    @property
    def t0(self):
        """
        The initial time set when the driver was created.
        """
        return self._t0

    def __repr__(self):
        return f"{type(self).__name__}(step_count={self.step_count}, t={self.t})"

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("generator     ", self._generator),
                ("integrator    ", self._integrator),
                ("linear solver ", self.linear_solver),
                ("State         ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)

    def _odefun(self, t, w, **kwargs):
        """
        Internal method which dispatches to the actual ODE system function.
        """
        return dwdt_mcstate(self.state, self, t, w, **kwargs)

    def _qgt_norm(self, x: PyTree):
        """
        Computes the norm induced by the QGT :math:`S`, i.e, :math:`x^\\dagger S x`.
        """
        y = self._last_qgt @ x
        xc_dot_y = nk.jax.tree_dot(nk.jax.tree_conj(x), y)
        return jnp.sqrt(jnp.real(xc_dot_y))
