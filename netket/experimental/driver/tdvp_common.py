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

from collections.abc import Callable, Sequence
from functools import partial
import warnings

import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

import netket as nk
from netket.driver import AbstractVariationalDriver
from netket.driver.abstract_variational_driver import _to_iterable
from netket.logging.json_log import JsonLog
from netket.operator import AbstractOperator
from netket.utils import mpi, timing
from netket.utils.dispatch import dispatch
from netket.utils.types import PyTree
from netket.utils.deprecation import warn_deprecation
from netket.vqs import VariationalState
from netket.experimental.dynamics import AbstractSolver, Integrator
from netket.experimental.dynamics._utils import (
    euclidean_norm,
    maximum_norm,
)


class TDVPBaseDriver(AbstractVariationalDriver):
    """
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :class:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.
    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        # TODO: remove default None once `integrator` is removed
        ode_solver: AbstractSolver = None,
        *,
        t0: float = 0.0,
        # TODO: integrator deprecated in 3.16 (oct/nov 2024)
        integrator: AbstractSolver = None,
        error_norm: str | Callable = "qgt",
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (Hamiltonian for pure states,
                Lindbladian for density operators).
            variational_state: The variational state.
            ode_solver: Solving algorithm used the ODE.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schrödinger equation (SE), "imag" for the imaginary-time SE.
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

        super().__init__(
            variational_state, optimizer=None, minimized_quantity_name="Generator"
        )

        self._generator_repr = repr(operator)
        if isinstance(operator, AbstractOperator):
            op = operator.collect()
            self._generator = lambda _: op
        else:
            self._generator = operator

        self._dw = None  # type: PyTree
        self._last_qgt = None
        self._integrator = None

        self.error_norm = error_norm

        if integrator is not None:
            warn_deprecation(
                "The keyword argument `integrator` has been renamed to `ode_solver` and "
                "deprecated to improve clarity.\n"
                "Update the keyword argument in your code to avoid breakage in the future."
            )
            if ode_solver is not None:
                raise ValueError("Cannot specify both `ode_solver` and `integrator`.")
            ode_solver = integrator
        if ode_solver is None:
            raise ValueError("You need to define the `ode_solver` for the TDVP driver.")
        self.ode_solver = ode_solver

        self._stop_count = 0
        self._postfix = {}

    @property
    def ode_solver(self):
        """
        The underlying solver which solves the ODE at each time step.
        """
        return self.integrator._solver

    @property
    def integrator(self):
        """
        The underlying integrator which computes the time steps.
        """
        return self._integrator

    @ode_solver.setter
    def ode_solver(self, new_ode_solver):
        if self._integrator is None:
            t0 = self.t0
        else:
            t0 = self.t

        self._integrator = Integrator(
            partial(odefun, self.state, self),
            new_ode_solver,
            t0,
            self.state.parameters,
            use_adaptive=new_ode_solver.adaptive,
            norm=self.error_norm,
            parameters=new_ode_solver.integrator_params,
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
    def error_norm(self, error_norm: str | Callable):
        if isinstance(error_norm, Callable):
            self._error_norm = error_norm
        elif error_norm == "euclidean":
            self._error_norm = euclidean_norm
        elif error_norm == "maximum":
            self._error_norm = maximum_norm
        elif error_norm == "qgt":
            self._error_norm = partial(qgt_norm, self)
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

    def iter(self, T: float, *, tstops: Sequence[float] | None = None):
        """
        Returns a generator which advances the time evolution for an interval
        of length :code:`T`, stopping at :code:`tstops`.

        Args:
            T: Length of the integration interval.
            tstops: A sequence of stopping times, each within the interval :code:`[self.t0, self.t0 + T]`,
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
        tstops: Sequence[float] | None = None,
        callback: Callable | None = None,
    ):
        """
        Implementation of :code:`iter`. This method accepts and additional `callback` object, which
        is called after every accepted step.
        """
        t_end = self.t + T
        if tstops is not None and (
            np.any(np.less(tstops, self.t)) or np.any(np.greater(tstops, t_end))
        ):
            raise ValueError(
                f"All tstops must be in range [t, t + T]=[{self.t}, {t_end}]"
            )

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
                        f"ODE integrator: {self._integrator.errors.message()}"
                    )
                elif self._integrator.warnings:
                    warnings.warn(
                        f"ODE integrator: {self._integrator.warnings.message()}",
                        UserWarning,
                        stacklevel=3,
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
        out=(),
        obs=None,
        *,
        tstops=None,
        show_progress=True,
        callback=None,
        timeit: bool = False,
    ):
        """
        Runs the time evolution.

        By default uses :class:`netket.logging.JsonLog`. To know about the output format
        check it's documentation. The logger object is also returned at the end of this function
        so that you can inspect the results without reading the json output.

        Args:
            T: The integration time period.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing the observables that should be computed.
            tstops: A sequence of stopping times, each within the interval :code:`[self.t0, self.t0 + T]`,
                at which the driver will stop and perform estimation of observables, logging, and execute
                the callback function. By default, a stop is performed after each time step (at potentially
                varying step size if an adaptive integrator is used).
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to be executed at each
                stopping time.
            timeit: If True, provide timing information.
        """
        if obs is None:
            obs = {}

        if callback is None:
            callback = lambda *_args, **_kwargs: True

        # if out is a path, create an overwriting Json Log for output
        if isinstance(out, str):
            out = JsonLog(out, "w")
        elif out is None:
            out = ()

        loggers = _to_iterable(out)
        callbacks = _to_iterable(callback)
        callback_stop = False

        t_end = np.asarray(self.t + T)

        with tqdm(
            total=t_end,
            disable=not show_progress or not self._is_root,
            unit_scale=True,
            dynamic_ncols=True,
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

                pbar.n = min(np.asarray(self._integrator.t), t_end)
                self._postfix["n"] = self.step_count
                self._postfix.update(
                    {
                        self._loss_name: str(self._loss_stats),
                    }
                )

                pbar.set_postfix(self._postfix)
                pbar.refresh()

            with timing.timed_scope(force=timeit) as timer:
                for step in self._iter(T, tstops=tstops, callback=update_progress_bar):
                    log_data = self.estimate(obs)
                    self._log_additional_data(log_data, step)

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

        if timeit:
            self._timer = timer
            if self._is_root:
                print(timer)
        return loggers

    def _log_additional_data(self, log_dict, step):
        super()._log_additional_data(log_dict, step)
        log_dict["t"] = self.t

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
        return odefun(self.state, self, t, w)


def qgt_norm(driver: TDVPBaseDriver, x: PyTree):
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
