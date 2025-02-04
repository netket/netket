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
from typing import TYPE_CHECKING

from netket.utils.struct import Pytree, field
from netket.utils.types import Callable, PyTree

from ._integrator_params import IntegratorParameters

if TYPE_CHECKING:
    from ._integrator_state import IntegratorState


class AbstractSolverState(Pytree):
    """
    Base class holding the state of a solver.
    """

    def __repr__(self):
        return "SolverState()"


class AbstractSolver(Pytree):
    r"""
    Abstract base class for ODE solvers.
    This object is an immutable pyTree. The structure used to hold any solver-specific data should be initialized by
    ``_init_state``.
    """

    dt: float = field(pytree_node=False)
    """The intial time-step size."""
    adaptive: bool = field(pytree_node=False, default=False)
    """The flag whether to use adaptive time-stepping."""
    integrator_params: IntegratorParameters = field(pytree_node=False)
    """Any additional arguments to pass to the Integrator"""

    def __init__(self, dt, adaptive=False, **kwargs):
        r"""
        Args:
            dt: The initial time-step size of the integrator.
            adaptive: A boolean indicator whether to use an adaptive scheme.

            atol: The tolerance for the absolute error on the solution if :code:`adaptive`.
                defaults to :code:`0.0`.
            rtol: The tolerance for the relative error on the solution if :code:`adaptive`.
                defaults to :code:`1e-7`.
            dt_limits: The extremal accepted values for the time-step size `dt` if :code:`adaptive`.
                defaults to :code:`(None, 10 * dt)`.
        """
        self.dt = dt
        self.adaptive = adaptive
        self.integrator_params = IntegratorParameters(dt=dt, **kwargs)

    def _init_state(self, integrator_state: "IntegratorState") -> AbstractSolverState:
        r"""
        Initializes the `SolverState` structure containing supplementary information needed.
        Args:
            integrator_state: The state of the Integrator

        Returns:
            An intialized `SolverState` instance
        """
        return AbstractSolverState()

    def step(
        self, f: Callable, dt: float, t: float, y_t: PyTree, state: AbstractSolverState
    ) -> tuple[PyTree, AbstractSolverState]:
        r"""
        Performs one fixed-size step from `t` to `t + dt`
        Args:
            f: The ODE function.
            dt: The current time-step size.
            t: The current time.
            y_t: The current solution.
            state: The state of the solver.

        Returns:
            The next solution y_t+1 and the corresponding updated state of the solver
        """

        raise NotImplementedError(
            "You need to define the method `step` in your `AbstractSolver`."
        )

    def step_with_error(
        self, f: Callable, dt: float, t: float, y_t: PyTree, state: AbstractSolverState
    ) -> tuple[PyTree, PyTree, AbstractSolverState]:
        r"""
        Perform one fixed-size step from `t` to `t + dt` and additionally returns the
        error vector provided by the adaptive solver.
        Args:
            f: The ODE function.
            dt: The current time-step size.
            t: The current time.
            y_t: The current solution.
            state: The state of the solver.

        Returns:
            The next solution y_t+1, the error y_err and the corresponding updated state of the solver
        """
        raise NotImplementedError(
            "You need to define the method `step_with_error` in your `AbstractSolver`."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dt={self.dt}, adaptive={self.adaptive}, integrator_parameters={self.integrator_params})"

    @property
    def is_explicit(self) -> bool:
        """Boolean indication whether the integrator is explicit."""
        raise NotImplementedError

    @property
    def is_adaptive(self) -> bool:
        """Boolean indication whether the integrator can be adaptive."""
        raise NotImplementedError

    @property
    def stages(self) -> int:
        """
        Number of stages (equal to the number of evaluations of the ode function) of the scheme.
        """
        raise NotImplementedError

    @property
    def is_fsal(self):
        """Returns True if the first iteration is the same as last."""
        # TODO: this is not yet supported
        return False


def append_docstring(doc):
    """
    Decorator that appends the string `doc` to the decorated function.

    This is needed here because docstrings cannot be f-strings or manipulated strings.
    """

    def _append_docstring(fun):
        fun.__doc__ = fun.__doc__ + doc
        return fun

    return _append_docstring


args_fixed_dt_docstring = """
    Args:
        dt: Timestep (floating-point number).
"""

args_adaptive_docstring = """
    This solver is adaptive, meaning that the time-step is changed at every
    iteration in order to keep the error below a certain threshold.

    In particular, given the variables at step :math:`t`, :math:`\\theta^{t}` and the
    error at the same time-step, :math:`\\epsilon^t`, we compute a rescaled error by
    using the absolute (**atol**) and relative (**reltol**) tolerances according
    to this formula.

    .. math::

        \\epsilon^\\text{scaled} = \\text{Norm}(\\frac{\\epsilon^{t}}{\\epsilon_{atol} +
            \\max(\\theta^t, \\theta^{t-1})\\epsilon_{reltol}}),

    where :math:`\\text{Norm}` is a function that normalises the vector, usually a vector
    norm but could be something else as well, and :math:`\\max` is an elementwise maximum
    function (with lexicographical ordering for complex numbers).

    Then, the integrator will attempt to keep `\\epsilon^\\text{scaled}<1`.

    Args:
        dt: Timestep (floating-point number). When :code:`adaptive==False` this value
            is never changed, when :code:`adaptive == True` this is the initial timestep.
        adaptive: Whether to use adaptive timestepping (Defaults to False).
            Not all integrators support adaptive timestepping.
        atol: Maximum absolute error at every time-step during adaptive timestepping.
            A larger value will lead to larger timestep. This option is ignored if
            `adaptive=False`. A value of 0 means it is ignored. Note that the `norm` used
            to compute the error can be  changed in the :ref:`netket.experimental.TDVP`
            driver. (Defaults to 0).
        rtol: Maximum relative error at every time-step during adaptive timestepping.
            A larger value will lead to larger timestep. This option is ignored if
            `adaptive=False`. Note that the `norm` used to compute the error can be
            changed in the :ref:`netket.experimental.TDVP` driver. (Defaults to 1e-7)
        dt_limits: A length-2 tuple of minimum and maximum timesteps considered by
            adaptive time-stepping. A value of None signals that there is no bound.
            Defaults to :code:`(None, 10*dt)`.
    """
