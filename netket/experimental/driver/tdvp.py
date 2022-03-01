# Copyright 2020, 2021  The NetKet Authors - All rights reserved.
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

from typing import Callable, Optional, Sequence, Union
import warnings

import jax
import jax.numpy as jnp
import jax.experimental.host_callback as hcb
import numpy as np
from tqdm import tqdm

import netket as nk
from netket.driver import AbstractVariationalDriver
from netket.driver.abstract_variational_driver import _to_iterable
from netket.driver.vmc_common import info
from netket.jax import HashablePartial
from netket.logging.json_log import JsonLog
from netket.operator import AbstractOperator
from netket.optimizer import LinearOperator
from netket.optimizer.qgt import QGTAuto
from netket.utils import mpi
from netket.utils.dispatch import dispatch
from netket.utils.types import PyTree
from netket.vqs import VariationalState, VariationalMixedState, MCState

from netket.experimental.dynamics import RKIntegratorConfig
from netket.experimental.dynamics._rk_solver import euclidean_norm, maximum_norm


class TDVP(AbstractVariationalDriver):
    """
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :ref:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.
    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        integrator: RKIntegratorConfig,
        *,
        t0: float = 0.0,
        propagation_type="real",
        qgt: LinearOperator = None,
        linear_solver=None,
        linear_solver_restart: bool = False,
        error_norm: Union[str, Callable] = "euclidean",
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (Hamiltonian for pure states,
                Lindbladian for density operators).
            variational_state: The variational state.
            integrator: Configuration of the algorithm used for solving the ODE.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schödinger equation (SE), "imag" for the imaginary-time SE.
            qgt: The QGT specification.
            linear_solver: The solver for solving the linear system determining the time evolution.
            linear_solver_restart: If False (default), the last solution of the linear system
                is used as initial value in subsequent steps.
            error_norm: Norm function used to calculate the error with adaptive integrators.
                Can be either "euclidean" for the standard L2 vector norm :math:`w^\dagger w`,
                "maximum" for the maximum norm :math:`\max_i |w_i|`
                or "qgt", in which case the scalar product induced by the QGT :math:`S` is used
                to compute the norm :math:`\Vert w \Vert^2_S = w^\dagger S w` as suggested
                in PRL 125, 100503 (2020).
                Additionally, it possible to pass a custom function with signature
                :code:`norm(x: PyTree) -> float`
                which maps a PyTree of parameters :code:`x` to the corresponding norm.
                Note that norm is used in jax.jit-compiled code.
        """
        self._t0 = t0

        if linear_solver is None:
            linear_solver = nk.optimizer.solver.svd
        if qgt is None:
            qgt = QGTAuto(solver=linear_solver)

        super().__init__(
            variational_state, optimizer=None, minimized_quantity_name="Generator"
        )

        self._generator_repr = repr(operator)
        if isinstance(operator, AbstractOperator):
            op = operator.collect()
            self._generator = lambda _: op
        else:
            self._generator = operator

        self.propagation_type = propagation_type
        if isinstance(variational_state, VariationalMixedState):
            # assuming Lindblad Dynamics
            # TODO: support density-matrix imaginary time evolution
            if propagation_type == "real":
                self._loss_grad_factor = 1.0
            else:
                raise ValueError(
                    "only real-time Lindblad evolution is supported for " "mixed states"
                )
        else:
            if propagation_type == "real":
                self._loss_grad_factor = -1.0j
            elif propagation_type == "imag":
                self._loss_grad_factor = -1.0
            else:
                raise ValueError("propagation_type must be one of 'real', 'imag'")

        self.qgt = qgt
        self.linear_solver = linear_solver
        self.linear_solver_restart = linear_solver_restart

        self._dw = None  # type: PyTree
        self._last_qgt = None
        self._integrator = None
        self._integrator_constructor = None

        self._odefun = HashablePartial(odefun_host_callback, self.state, self)

        self.error_norm = error_norm
        self.integrator = integrator

        self._stop_count = 0
        self._postfix = {}

    @property
    def integrator(self):
        """
        The underlying integrator which computes the time steps.
        """
        return self._integrator

    @integrator.setter
    def integrator(self, integrator):
        if self._integrator is None:
            t0 = self.t0
        else:
            t0 = self.t

        self._integrator_constructor = integrator

        self._integrator = integrator(
            self._odefun,
            t0,
            self.state.parameters,
            norm=self.error_norm,
        )

    @property
    def generator(self) -> Callable:
        """
        The generator of the dynamics as a function with signature
            generator(t: float) -> AbstractOperator
        """
        return self._generator

    @property
    def error_norm(self) -> Callable:
        """
        Returns the Callable function computing the error of the norm used for adaptive
        timestepping by the integrator.

        Can be set to a Callable accepting a pytree and returning a real scalar, or
        a string between 'euclidean', 'maximum' or 'qgt'.
        """
        return self._error_norm

    @error_norm.setter
    def error_norm(self, error_norm: Union[str, Callable]):
        if isinstance(error_norm, Callable):
            self._error_norm = error_norm
        elif error_norm == "euclidean":
            self._error_norm = euclidean_norm
        elif error_norm == "maximum":
            self._error_norm = maximum_norm
        elif error_norm == "qgt":
            w = self.state.parameters
            norm_dtype = nk.jax.dtype_real(nk.jax.tree_dot(w, w))
            # QGT norm is called via host callback since it accesses the driver
            # TODO: make this also an hashablepartial on self to reduce recompilation
            self._error_norm = lambda x: hcb.call(
                HashablePartial(qgt_norm, self),
                x,
                result_shape=jax.ShapeDtypeStruct((), norm_dtype),
            )
        else:
            raise ValueError(
                "error_norm must be a callable or one of 'euclidean', 'qgt', 'maximum',"
                f" but {error_norm} was passed."
            )

        if self.integrator is not None:
            self.integrator.norm = self._error_norm

    def advance(self, T: float):
        """
        Advance the time propagation by :code:`T` to :code:`self.t + T`.

        Args:
            T: Length of the integration interval.
        """
        for _ in self.iter(T):
            pass

    def iter(self, T: float, *, tstops: Optional[Sequence[float]] = None):
        """
        Returns a generator which advances the time evolution for an interval
        of length :code:`T`, stopping at :code:`tstops`.

        Args:
            T: Length of the integration interval.
            tstops: A sequence of stopping times, each within the intervall :code:`[self.t0, self.t0 + T]`,
                at which this method will stop and yield. By default, a stop is performed
                after each time step (at potentially varying step size if an adaptive
                integrator is used).
        Yields:
            The current step count.
        """
        yield from self._iter(T, tstops)

    def _iter(
        self,
        T: float,
        tstops: Optional[Sequence[float]] = None,
        callback: Callable = None,
    ):
        """
        Implementation of :code:`iter`. This method accepts and additional `callback` object, which
        is called after every accepted step.
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
                if self._integrator.errors:
                    raise RuntimeError(
                        f"RK solver: {self._integrator.errors.message()}"
                    )
                elif self._integrator.warnings:
                    warnings.warn(
                        f"RK solver: {self._integrator.warnings.message()}",
                        UserWarning,
                    )
            self._step_count += 1
            # optionally call callback
            if callback:
                callback()

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
        Runs the time evolution.

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
            if out is None:
                loggers = ()
            # if out is a path, create an overwriting Json Log for output
            elif isinstance(out, str):
                loggers = (JsonLog(out, "w"),)
            else:
                loggers = _to_iterable(out)
        else:
            loggers = tuple()
            show_progress = False

        callbacks = _to_iterable(callback)
        callback_stop = False

        with tqdm(
            total=np.asarray(self.t + T),
            disable=not show_progress,
            unit_scale=True,
        ) as pbar:
            first_step = True

            # We need a closure to pass to self._iter in order to update the progress bar even if
            # there are no tstops
            def update_progress_bar():
                # Reset the timing of tqdm after the first step to ignore compilation time
                nonlocal first_step
                if first_step:
                    first_step = False
                    pbar.unpause()

                pbar.n = np.asarray(self._integrator.t)
                self._postfix["n"] = self.step_count
                pbar.set_postfix(self._postfix)
                pbar.refresh()

            for step in self._iter(T, tstops=tstops, callback=update_progress_bar):
                log_data = self.estimate(obs)

                self._postfix = {"n": self.step_count}
                # if the cost-function is defined then report it in the progress bar
                if self._loss_stats is not None:
                    self._postfix.update(
                        {
                            self._loss_name: str(self._loss_stats),
                        }
                    )
                    log_data[self._loss_name] = self._loss_stats
                pbar.set_postfix(self._postfix)

                # Execute callbacks before loggers because they can append to log_data
                for callback in callbacks:
                    if not callback(step, log_data, self):
                        callback_stop = True

                for logger in loggers:
                    logger(self.step_value, log_data, self.state)

                if len(callbacks) > 0:
                    if mpi.mpi_any(callback_stop):
                        break
                update_progress_bar()

            # Final update so that it shows up filled.
            update_progress_bar()

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
        """Current time step."""
        return self._integrator.dt

    @property
    def t(self):
        """Current time."""
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
                ("generator     ", self._generator_repr),
                ("integrator    ", self._integrator),
                ("linear solver ", self.linear_solver),
                ("state         ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)

    def ode(self, t=None, w=None):
        r"""
        Evaluates the TDVP equation of motion

        .. math::

            G(w) \dot w = \gamma F(w, t)

        where :math:`G(w)` is the QGT, :math:`F(w, t)` the gradient of :code:`self.generator`
        and :math:`\gamma` one of
        :math:`\gamma = -1` (imaginary-time dynamics for :code:`MCState`),
        :math:`\gamma = -i` (real-time dynamics for :code:`MCState`), or
        :math:`\gamma = 1` (real-time dynamics for :code:`MCMixedState`).

        Args:
            t: Time (defaults to :code:`self.t`).
            w: Variational parameters (defaults to :code:`self.state.parameters`).

        Returns:
            The time-derivative :math:`\dot w`.
        """
        if t is None:
            t = self.t
        if w is None:
            w = self.state.parameters
        return self._odefun(t, w)


def qgt_norm(driver: TDVP, x: PyTree):
    """
    Computes the norm induced by the QGT :math:`S`, i.e, :math:`x^\\dagger S x`.
    """
    y = driver._last_qgt @ x  # pylint: disable=protected-access
    xc_dot_y = nk.jax.tree_dot(nk.jax.tree_conj(x), y)
    return jnp.sqrt(jnp.real(xc_dot_y))


@dispatch
def odefun(state, driver, t, w, **kwargs):
    # pylint: disable=unused-argument
    raise NotImplementedError(f"odefun not implemented for {type(state)}")


@dispatch
def odefun(state: MCState, driver: TDVP, t, w, *, stage=0):  # noqa: F811
    # pylint: disable=protected-access

    state.parameters = w
    state.reset()

    driver._loss_stats, driver._loss_grad = state.expect_and_grad(
        driver.generator(t),
        use_covariance=True,
    )
    driver._loss_grad = jax.tree_map(
        lambda x: driver._loss_grad_factor * x, driver._loss_grad
    )

    qgt = driver.qgt(driver.state)
    if stage == 0:  # TODO: This does not work with FSAL.
        driver._last_qgt = qgt

    initial_dw = None if driver.linear_solver_restart else driver._dw
    driver._dw, _ = qgt.solve(driver.linear_solver, driver._loss_grad, x0=initial_dw)
    return driver._dw


def odefun_host_callback(state, driver, *args, **kwargs):
    """
    Calls odefun through a host callback in order to make the rest of the
    ODE solver jit-able.
    """
    result_shape = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        state.parameters,
    )

    return hcb.call(
        lambda args_and_kw: odefun(state, driver, *args_and_kw[0], **args_and_kw[1]),
        # pack args and kwargs together, since host_callback passes a single argument:
        (args, kwargs),
        result_shape=result_shape,
    )
