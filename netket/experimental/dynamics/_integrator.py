# Copyright 2021 The NetKet Authors - All Rights Reserved.
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

from collections.abc import Callable

import jax
import jax.numpy as jnp

from netket.utils.numbers import dtype as _dtype
from netket.utils.mpi.primitives import mpi_all_jax
from netket.utils import struct
from netket.utils.types import PyTree

from ._integrator_state import IntegratorState, IntegratorFlags
from ._integrator_params import IntegratorParameters
from ._solver import AbstractSolver
from ._utils import (
    scaled_error,
    propose_time_step,
    set_flag_jax,
    euclidean_norm,
    maximum_norm,
)


class Integrator(struct.Pytree, mutable=True):
    r"""
    Ordinary-Differential-Equation integrator.
    Given an ODE-function :math:`dy/dt = f(t, y)`, it integrates the derivatives to obtain the solution
    at the next time step :math:`y_{t+1}`.
    """

    f: Callable = struct.field(pytree_node=False)
    """The ODE function."""

    _state: IntegratorState
    """The state of the integrator, containing informations about the solution."""
    _solver: AbstractSolver
    """The ODE solver."""
    _parameters: IntegratorParameters
    """The options of the integration."""

    use_adaptive: bool = struct.field(pytree_node=False)
    """Boolean indicating whether to use an adaptative scheme."""
    norm: Callable = struct.field(pytree_node=False)
    """The norm used to estimate the error."""

    def __init__(
        self,
        f: Callable,
        solver: AbstractSolver,
        t0: float,
        y0: PyTree,
        use_adaptive: bool,
        parameters: IntegratorParameters,
        norm: str | Callable = None,
    ):
        r"""
        Args:
            f: The ODE function
                Given a time `t` and a state `y_t`, it should return the partial
                derivatives in the same format as `y_t`. The function should also accept
                supplementary arguments, such as :code:`stage`.
            solver: The ODE solver.
                If :code:`use_adaptive`, it should define the method :code:`step_with_error(f,dt,t,y_t,solver_state)`
                otherwise, the method :code:`step(f,dt,t,y_t,solver_state)`
            t0: The intial time.
            y0: The initial state.
            use_adaptive: The (boolean) indicator whether an adaptive (time-step) scheme is used.
            parameters: The suppleementary hyper-parameters of the integrator.
                This includes the values for :code:`dt`, :code:`atol`, :code`rtol` and :code`dt_limits`
                See :code:`IntegratorParameters` for more details.
            norm: The function used for the norm of the error.
                By default, we use euclidean_norm.
        """
        self.f = f
        self._solver = solver
        self._parameters = parameters

        self.use_adaptive = use_adaptive

        if norm is not None:
            if isinstance(norm, str):
                norm = norm.lower()
                if norm == "euclidean":
                    norm = euclidean_norm
                elif norm == "maximum":
                    norm = maximum_norm
                else:
                    raise ValueError(
                        f"The error norm must either be 'euclidean' or 'maximum', instead got {norm}."
                    )
            if not isinstance(norm, Callable):
                raise ValueError(
                    f"The error norm must be a callable, instead got a {type(norm)}."
                )
        else:
            norm = euclidean_norm
        self.norm = norm

        self._state = self._init_state(t0, y0)

    def _init_state(self, t0: float, y0: PyTree) -> IntegratorState:
        r"""
        Initializes the `IntegratorState` structure containing the solver and state,
        given the necessary information.

        Args:
            t0: The initial time of evolution
            y0: The solution at initial time `t0`

        Returns:
            An :code:`Integrator` instance intialized with the passed arguments
        """
        dt = self._parameters.dt

        t_dtype = jnp.result_type(_dtype(t0), _dtype(dt))

        return IntegratorState(
            t=jnp.array(t0, dtype=t_dtype),
            y=y0,
            dt=jnp.array(dt, dtype=t_dtype),
            solver=self._solver,
            last_norm=0.0 if self.use_adaptive else None,
            last_scaled_error=0.0 if self.use_adaptive else None,
            flags=IntegratorFlags(0),
        )

    def step(self, max_dt: float = None) -> bool:
        """
        Performs one full step by :code:`min(self.dt, max_dt)`.

        Returns:
            A boolean indicating whether the step was successful or
            was rejected by the step controller and should be retried.

            Note that the step size can be adjusted by the step controller
            in both cases, so the integrator state will have changed
            even after a rejected step.
        """
        if not self.use_adaptive:
            self._state = self._step_fixed(
                solver=self._solver,
                f=self.f,
                state=self._state,
                max_dt=max_dt,
                parameters=self._parameters,
            )
        else:
            self._state = self._step_adaptive(
                solver=self._solver,
                f=self.f,
                state=self._state,
                max_dt=max_dt,
                parameters=self._parameters,
                norm_fn=self.norm,
            )

        return self._state.accepted

    @property
    def t(self) -> float:
        """The actual time."""
        return self._state.t.value

    @property
    def y(self) -> PyTree:
        """The actual state."""
        return self._state.y

    @property
    def dt(self) -> float:
        """The actual time-step size."""
        return self._state.dt

    @property
    def solver(self) -> AbstractSolver:
        """The ODE solver."""
        return self._solver

    def _get_integrator_flags(self, intersect=IntegratorFlags.NONE) -> IntegratorFlags:
        r"""Returns the currently set flags of the integrator, intersected with `intersect`."""
        # _state.flags is turned into an int-valued DeviceArray by JAX,
        # so we convert it back.
        return IntegratorFlags(int(self._state.flags) & intersect)

    @property
    def errors(self) -> IntegratorFlags:
        r"""Returns the currently set error flags of the integrator."""
        return self._get_integrator_flags(IntegratorFlags.ERROR_FLAGS)

    @property
    def warnings(self) -> IntegratorFlags:
        r"""Returns the currently set warning flags of the integrator."""
        return self._get_integrator_flags(IntegratorFlags.WARNINGS_FLAGS)

    def __repr__(self) -> str:
        return "{}(solver={}, state={}, adaptive={}{})".format(
            "Integrator",
            self.solver,
            self._state,
            self.use_adaptive,
            (f", norm={self.norm}" if self.use_adaptive else ""),
        )

    @staticmethod
    def _step_fixed(
        solver: AbstractSolver,
        f: Callable,
        state: IntegratorState,
        max_dt: float | None,
        parameters: IntegratorParameters,
    ) -> IntegratorState:
        r"""
        Performs one fixed step from current time.
        Args:
            solver: Instance that solves the ODE.
                The solver should contain a method :code:`step(f,dt,t,y_t,solver_state)`
            f: A callable ODE function.
                Given a time `t` and a state `y_t`, it should return the partial
                derivatives in the same format as `y_t`. The function should also accept
                supplementary arguments, such as :code:`stage`.
            state: IntegratorState containing the current state (t,y), the solver_state and stability information.
            max_dt: The maximal value for the time step `dt`.
            parameters: The integration parameters.

        Returns:
            Updated state of the integrator.
        """
        del parameters

        if max_dt is None:
            actual_dt = state.dt
        else:
            actual_dt = jnp.minimum(state.dt, max_dt)

        # Perform the solving step
        y_tp1, solver_state = solver.step(
            f, actual_dt, state.t.value, state.y, state.solver_state
        )

        return state.replace(
            step_no=state.step_no + 1,
            step_no_total=state.step_no_total + 1,
            t=state.t + actual_dt,
            y=y_tp1,
            solver_state=solver_state,
            flags=IntegratorFlags.INFO_STEP_ACCEPTED,
        )

    @staticmethod
    def _step_adaptive(
        solver: AbstractSolver,
        f: Callable,
        state: IntegratorState,
        max_dt: float | None,
        parameters: IntegratorParameters,
        norm_fn: Callable,
    ) -> IntegratorState:
        r"""
        Performs one adaptive step from current time.
        Args:
            solver: Instance that solves the ODE
                The solver should contain a method :code:`step_with_error(f,dt,t,y_t,solver_state)`
            f: A callable ODE function.
                Given a time `t` and a state `y_t`, it should return the partial
                derivatives in the same format as `y_t`. The function should also accept
                supplementary arguments, such as :code:`stage`.
            state: IntegratorState containing the current state (t,y), the solver_state and stability information.
            norm_fn: The function used for the norm of the error.
                By default, we use euclidean_norm.
            parameters: The integration parameters.
            max_dt: The maximal value for the time-step size `dt`.

        Returns:
            Updated state of the integrator
        """
        flags = IntegratorFlags(0)

        if max_dt is None:
            actual_dt = state.dt
        else:
            actual_dt = jnp.minimum(state.dt, max_dt)

        # Perform the solving step
        y_tp1, y_err, solver_state = solver.step_with_error(
            f, actual_dt, state.t.value, state.y, state.solver_state
        )

        scaled_err, norm_y = scaled_error(
            y_tp1,
            y_err,
            parameters.atol,
            parameters.rtol,
            last_norm_y=state.last_norm,
            norm_fn=norm_fn,
        )

        # Propose the next time step, but limited within [0.1 dt, 5 dt] and potential
        # global limits in dt_limits. Not used when actual_dt < state.dt (i.e., the
        # integrator is doing a smaller step to hit a specific stop).
        dt_min, dt_max = parameters.dt_limits
        next_dt = propose_time_step(
            actual_dt,
            scaled_err,
            solver.error_order,
            limits=(
                (jnp.maximum(0.1 * state.dt, dt_min) if dt_min else 0.1 * state.dt),
                (jnp.minimum(5.0 * state.dt, dt_max) if dt_max else 5.0 * state.dt),
            ),
        )

        # check if next dt is NaN
        flags = set_flag_jax(
            ~jnp.isfinite(next_dt), flags, IntegratorFlags.ERROR_INVALID_DT
        )

        # check if we are at lower bound for dt
        if dt_min is not None:
            is_at_min_dt = jnp.isclose(next_dt, dt_min)
            flags = set_flag_jax(is_at_min_dt, flags, IntegratorFlags.WARN_MIN_DT)
        else:
            is_at_min_dt = False
        if dt_max is not None:
            is_at_max_dt = jnp.isclose(next_dt, dt_max)
            flags = set_flag_jax(is_at_max_dt, flags, IntegratorFlags.WARN_MAX_DT)

        # accept if error is within tolerances or we are already at the minimal step
        accept_step = jnp.logical_or(scaled_err < 1.0, is_at_min_dt)
        # accept the time step iff it is accepted by all MPI processes
        accept_step, _ = mpi_all_jax(accept_step)

        return jax.lax.cond(
            accept_step,
            # step accepted
            lambda _: state.replace(
                step_no=state.step_no + 1,
                step_no_total=state.step_no_total + 1,
                y=y_tp1,
                t=state.t + actual_dt,
                dt=jax.lax.cond(
                    actual_dt == state.dt,
                    lambda _: next_dt,
                    lambda _: state.dt,
                    None,
                ),
                last_norm=norm_y.astype(state.last_norm.dtype),
                last_scaled_error=scaled_err.astype(state.last_scaled_error.dtype),
                solver_state=solver_state,
                flags=flags | IntegratorFlags.INFO_STEP_ACCEPTED,
            ),
            # step rejected, repeat with lower dt
            lambda _: state.replace(
                step_no_total=state.step_no_total + 1,
                dt=next_dt,
                flags=flags,
            ),
            state,
        )
