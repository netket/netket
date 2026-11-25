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

from typing import Any
from collections.abc import Callable
from copy import copy
from functools import partial
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import netket as nk
from netket.operator import AbstractOperator
from netket.utils import struct
from netket.utils.dispatch import dispatch
from netket.utils.types import PyTree
from netket.vqs import VariationalState
from netket.experimental.dynamics import AbstractSolver, Integrator
from netket.experimental.dynamics._utils import (
    euclidean_norm,
    maximum_norm,
)

from netket._src.callbacks.base import (
    AbstractCallback,
    STOPPING_CALLBACK_ORDER,
    RunFinished,
)
from netket._src.callbacks.callback_list import CallbackList
from netket._src.callbacks.legacy_wrappers import (
    LegacyCallbackWrapper,
    LegacyLoggerWrapper,
)
from netket._src.callbacks.observables import ObservableCallback
from netket._src.callbacks.progressbar import TimeProgressBarCallback
from netket._src.driver.abstract_variational_driver import AbstractDriver


class TDVPBaseDriver(AbstractDriver):
    """
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :class:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.
    """

    _generator: Callable = struct.field(pytree_node=False, serialize=False)
    _t0: float = struct.field(pytree_node=False, serialize=False)
    _error_norm: Callable | str = struct.field(pytree_node=False, serialize=False)

    # State of the time evolution: the integrator holds the time, the time step
    # and the parameters, and `_dw` is the last solution of the equation of
    # motion, used as the starting point of the next one.
    _integrator: Integrator
    _dw: PyTree

    # Recomputed at every step
    _last_qgt: Any = struct.field(serialize=False, default=None)

    # The times at which the current (or last) call to `run` started and stops,
    # and the stopping times it has not reached yet (None to stop at every step).
    _t_start: Any = struct.field(pytree_node=False, serialize=False, default=None)
    _t_end: Any = struct.field(pytree_node=False, serialize=False, default=None)
    _tstops: np.ndarray | None = struct.field(
        pytree_node=False, serialize=False, default=None
    )
    # The current step: its time if it is a stopping time (None otherwise), the
    # longest it can be to not go past the next stopping time, and whether it is
    # the last step of the run, which only reports the final time.
    _stop_time: Any = struct.field(pytree_node=False, serialize=False, default=None)
    _max_dt: Any = struct.field(pytree_node=False, serialize=False, default=None)
    _is_last_step: bool = struct.field(
        pytree_node=False, serialize=False, default=False
    )

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        ode_solver: AbstractSolver,
        *,
        t0: float = 0.0,
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

        super().__init__(variational_state, minimized_quantity_name="Generator")

        if isinstance(operator, AbstractOperator):
            op = operator.collect()
            self._generator = lambda _: op
        else:
            self._generator = operator

        self._dw = jax.tree.map(jnp.zeros_like, self.state.parameters)
        self._integrator = None

        self.error_norm = error_norm
        self.ode_solver = ode_solver

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
        if isinstance(self._error_norm, str):
            return partial(qgt_norm, self)
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
            # Bound to the driver when it is used, as this one may be copied
            # (e.g. when restoring it from a checkpoint).
            self._error_norm = "qgt"
        else:
            raise ValueError(
                "error_norm must be a callable or one of 'euclidean', 'qgt', 'maximum',"
                f" but {error_norm} was passed."
            )

        if self.integrator is not None:
            self.integrator.norm = self.error_norm

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
                stopping time, with the time as their first argument. Like the loggers and the
                observables, they are not called at the other time steps.
                :class:`~netket.callbacks.AbstractCallback` objects are instead called at every
                time step, with the step count, as in the other drivers.
            timeit: If True, provide timing information.
        """
        # Fixed before `on_run_start`, so that a callback restoring the time there
        # (e.g. when resuming from a checkpoint) does not move the end of the run.
        self._t_start = self.t
        self._t_end = self.t + T
        self._tstops = _check_tstops(tstops, self._t_start, self._t_end)
        # The run ends at the final time, not after a number of steps.
        return super().run(
            sys.maxsize,
            out=out,
            obs=obs,
            show_progress=show_progress,
            callback=callback,
            timeit=timeit,
        )

    def compute_loss_and_update(self):
        """
        Attempts a step of the integrator from the current time, without changing
        the time and the parameters of the driver.

        Returns the value of the generator at the start of the step, and the
        integrator after the step, which :meth:`update_parameters` applies.
        """
        if self._step_attempt == 0:
            at_stop, self._max_dt, self._is_last_step, self._tstops = _next_step(
                self.t, self._t_end, self._tstops
            )
            self._stop_time = self.t if at_stop else None
        if self._is_last_step:
            return None, None

        loss_stats = None

        def f(t, w, *, stage=0):
            nonlocal loss_stats
            dw = odefun(self.state, self, t, w, stage=stage)
            if stage == 0:
                loss_stats = self._loss_stats
            return dw

        # Bound to this driver, which may be a copy of the one that created the
        # integrator (e.g. when restoring it from a checkpoint).
        integrator = copy(self._integrator)
        integrator.f = f
        integrator.norm = self.error_norm
        integrator.step(max_dt=self._max_dt)

        # The ODE function sets the parameters of the intermediate stages.
        self.state.parameters = self._integrator.y

        if integrator.errors:
            raise RuntimeError(f"ODE integrator: {integrator.errors.message()}")
        elif integrator.warnings:
            warnings.warn(
                f"ODE integrator: {integrator.warnings.message()}",
                UserWarning,
                stacklevel=2,
            )
        return loss_stats, integrator

    def _accept_step(self) -> bool:
        integrator = self._dp
        if integrator is None or integrator._state.accepted:
            return True
        # The step controller has changed the time step: retry with it.
        self._integrator = integrator
        return False

    def update_parameters(self, dp):
        """
        Moves the time and the parameters to the end of the step computed by
        :meth:`compute_loss_and_update`.

        Args:
            dp: the integrator after the step.
        """
        if dp is not None:
            self._integrator = dp
            self.state.parameters = dp.y

    def _default_callbacks(
        self,
        callbacks,
        *,
        n_iter,
        obs,
        step_size,
        loggers,
        show_progress,
        **kwargs,
    ) -> CallbackList:
        # Observables, loggers and callback functions are only called at the
        # stopping times, with the time instead of the step count.
        callback_list = []
        if obs is not None:
            # First, so that callbacks find the observables in the logged data.
            callback_list.append(_AtStoppingTimes(ObservableCallback(obs, interval=0)))
        for callback in callbacks:
            if isinstance(callback, AbstractCallback):
                callback_list.append(callback)
            else:
                callback_list.append(_AtStoppingTimes(LegacyCallbackWrapper(callback)))
        for log in loggers:
            if not isinstance(log, AbstractCallback):
                log = LegacyLoggerWrapper(log)
            callback_list.append(_AtStoppingTimes(log))
        if show_progress:
            callback_list.append(
                TimeProgressBarCallback(float(self._t_end - self._t_start))
            )
        callback_list.append(_StopAfterLastStep())
        return CallbackList(callback_list)

    def _log_additional_data(self, log_dict):
        super()._log_additional_data(log_dict)
        log_dict["t"] = self.t

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


def _check_tstops(tstops, t_start, t_end) -> np.ndarray | None:
    """
    Checks that the stopping times are within the run, and sorts them.
    Returns None, meaning to stop at every step, if there are none.
    """
    if tstops is None or len(tstops) == 0:
        return None
    if np.any(np.less(tstops, t_start)) or np.any(np.greater(tstops, t_end)):
        raise ValueError(f"All tstops must be in range [t, t + T]=[{t_start}, {t_end}]")
    return np.sort(tstops)


def _next_step(t, t_end, tstops):
    """
    Plans the step of a run ending at `t_end` that starts at time `t`.

    Returns whether `t` is a stopping time, the longest the step can be to not
    go past the next stopping time (None if unbounded), whether it is the last
    step of the run, which only reports the final time, and the stopping times
    after `t`. `tstops` is None to stop at every step.
    """
    is_last_step = not t < t_end
    if tstops is None:
        at_stop = np.isclose(t, t_end) if is_last_step else True
        return at_stop, None, is_last_step, None

    # The stopping times before `t` have already been reached, or skipped if a
    # callback moved the time forward (e.g. when resuming from a checkpoint).
    t_np = np.asarray(t)
    reached = (tstops < t_np) | np.isclose(tstops, t_np)
    at_stop = bool(np.any(np.isclose(tstops[reached], t_np)))
    tstops = tstops[~reached]

    if len(tstops) > 0 and not is_last_step:
        max_dt = tstops[0] - t
    else:
        max_dt = None
    return at_stop, max_dt, is_last_step, tstops


class _AtStoppingTimes(AbstractCallback):
    """
    Calls a callback or a logger only at the stopping times of
    :meth:`TDVPBaseDriver.run`, with the time instead of the step count.
    """

    callback: AbstractCallback

    def __init__(self, callback: AbstractCallback):
        self.callback = callback

    @property
    def callback_order(self) -> int:
        return self.callback.callback_order

    def on_run_start(self, step, driver):
        self.callback.on_run_start(driver.t, driver)

    def before_parameter_update(self, step, log_data, driver):
        if driver._stop_time is not None:
            self.callback.before_parameter_update(driver._stop_time, log_data, driver)

    def on_step_end(self, step, log_data, driver):
        if driver._stop_time is not None:
            self.callback.on_step_end(driver._stop_time, log_data, driver)

    def on_run_end(self, step, driver):
        self.callback.on_run_end(driver.t, driver)

    def on_run_error(self, step, error, driver):
        self.callback.on_run_error(driver.t, error, driver)


class _StopAfterLastStep(AbstractCallback):
    """
    Ends :meth:`TDVPBaseDriver.run` after its last step, which only reports the
    final time.
    """

    @property
    def callback_order(self) -> int:
        # Run last, so raising StopRun never skips a later callback's collective.
        return STOPPING_CALLBACK_ORDER

    def on_step_end(self, step, log_data, driver):
        if driver._is_last_step:
            raise RunFinished(f"Reached t={driver.t:.6g}")


@dispatch
def odefun(state, driver, t, w, **kwargs):
    # pylint: disable=unused-argument
    raise NotImplementedError(f"odefun not implemented for {type(state)}")
