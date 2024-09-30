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

from netket.utils.struct import Pytree, field
from netket.utils.types import Any, Callable
from abc import abstractmethod


from ._state import IntegratorState

SolverState = Any


class AbstractSolver(Pytree):
    r"""
    The ODE solver. Given the ODE :math:`dy/dt = F(t, y)`, it finds the solution :math:`y(t)`.
    Also works as a constructor for the `SolverState` instance if required.
    """

    initial_dt: float = field(pytree_node=False)
    """The intial time-step size."""
    adaptive: bool = field(pytree_node=False, default=False)
    """The flag whether to use adaptive time-stepping."""
    kwargs: Any = field(pytree_node=False)
    """Any additional arguments to pass to the Integrator"""

    def __init__(self, dt, *, adaptive=False, **kwargs):
        r"""
        Args:
            dt: The initial time-step size of the solver.
            adaptive: A boolean indicator whether to use an adaptive scheme.
        """
        self.initial_dt = dt
        self.adaptive = adaptive
        self.kwargs = kwargs

    def _init_state(self, integrator_state: IntegratorState) -> SolverState:
        r"""
        Initializes the `SolverState` structure containing supplementary information needed.
        Args:
            integrator_state: The state of the Integrator

        Returns:
            An intialized `SolverState` instance
        """
        return None

    @abstractmethod
    def step(
        self, f: Callable, dt: float, t: float, y_t: Pytree, state: SolverState
    ) -> tuple[Pytree, SolverState]:
        r"""
        Performs one fixed-size step from `t` to `t + dt`
        Args:
            f: The ODE function
            dt: The current time-step size
            t: The current time
            y_t: The current solution
            state: The state of the solver

        Returns:
            The next solution y_t+1 and the corresponding updated state of the solver

        """

        raise NotImplementedError(
            "You need to define the method `step` in your `AbstractSolver`."
        )

    @abstractmethod
    def step_with_error(
        self, f: Callable, dt: float, t: float, y_t: Pytree, state: SolverState
    ) -> tuple[Pytree, Pytree, SolverState]:
        r"""
        Perform one fixed-size step from `t` to `t + dt` and additionally returns the
        error vector provided by the adaptive solver.
        Args:
            f: The ODE function
            dt: The current time-step size
            t: The current time
            y_t: The current solution
            state: The state of the solver

        Returns:
            The next solution y_t+1, the error y_err and the corresponding updated state of the solver
        """
        raise NotImplementedError(
            "You need to define the method `step_with_error` in your `AbstractSolver`."
        )

    def __repr__(self):
        return "{}(dt={}, adaptive={}{})".format(
            self.__class__.__name__,
            self.initial_dt,
            self.adaptive,
            f", **kwargs={self.kwargs}" if self.kwargs else "",
        )

    @property
    @abstractmethod
    def is_explicit(self):
        """Boolean indication whether the integrator is explicit."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_adaptive(self):
        """Boolean indication whether the integrator can be adaptive."""
        raise NotImplementedError

    @property
    @abstractmethod
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function) of the scheme.
        """
        raise NotImplementedError
