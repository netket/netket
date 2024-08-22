# Copyright 2021 The NetKet Authors - All rights reserved.
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

import abc

from netket.utils.types import PyTree, Scalar
from netket.utils import timing, struct
from netket.vqs import VariationalState

from .linear_operator import LinearOperator, SolverT

# Generic signature of a preconditioner function/object

PreconditionerT = Callable[[VariationalState, PyTree, Scalar | None], PyTree]
"""Signature for Gradient preconditioners supported by NetKet drivers."""

LHSConstructorT = Callable[[VariationalState, Scalar | None], LinearOperator]
"""Signature for the constructor of a LinerOperator"""


class IdentityPreconditioner(struct.Pytree):
    """
    A preconditioner that does not transform the gradient.
    """

    def __call__(
        self,
        vstate: VariationalState,
        gradient: PyTree,
        step: Scalar | None = 0,
        *args,
        **kwargs,
    ) -> PyTree:
        return gradient


# For backward compatibility reasons
identity_preconditioner = IdentityPreconditioner()


class AbstractLinearPreconditioner(struct.Pytree, mutable=True):
    """Base class for a Linear Preconditioner solving a system :math:`Sx = F`.

    A LinearPreconditioner modifies the gradient :math:`F` in such a way that the new
    gradient :math:`x` solves the linear system `:math:`Sx=F`. The linear operator
    :math:`S` is constructed from the variational state.

    To subtype this class and provide a concrete implementation, one needs to define
    at least the function

    .. code::

        @dataclass
        class MyLinearPreconditioner(AbstractLinearPreconditioner):

            def lhs_constructor(self, vstate: VariationalState, step: Optional[Scalar] = 0):
                # here the lhs of the system should be constructed, for example by
                # returning the geometric tensor or any other object
                # return vstate.quantum_geometric_tensor()

    """

    solver: SolverT = struct.field(serialize=False)
    """Function used to solve the linear system."""

    solver_restart: bool = False
    """If False uses the last solution of the linear system as a starting point for the solution
    of the next."""

    x0: PyTree | None = None
    """Solution of the last linear system solved."""

    info: Any = struct.field(serialize=False, default=None)
    """Additional information returned by the solver when solving the last linear system."""

    _lhs: LinearOperator = struct.field(serialize=False, default=None)
    """LHS of the last linear system solved."""

    def __init__(self, solver, *, solver_restart=False):
        """
        Constructs the structure holding the parameters for using the
        linear preconditioner.

        Args:
            solver: A callable that solves a linear system of equations.
            solver_restart: If False uses the last solution of the linear
                system as a starting point for the solution of the next
                (default=False).
        """
        self.solver = solver
        self.solver_restart = solver_restart

    @timing.timed
    def __call__(
        self,
        vstate: VariationalState,
        gradient: PyTree,
        step: Scalar | None = None,
        *args,
        **kwargs,
    ) -> PyTree:
        self._lhs = self.lhs_constructor(vstate, step)

        x0 = self.x0 if self.solver_restart else None
        self.x0, self.info = self._lhs.solve(self.solver, gradient, x0=x0)

        return self.x0

    @abc.abstractmethod
    def lhs_constructor(self, vstate: VariationalState, step: Scalar | None = None):
        """
        This method should construct the left hand side of the linear system,
        which should be a linear operator.
        """

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"\n\tsolver          = {self.solver}, "
            + f"\n\tsolver_restart  = {self.solver_restart},"
            + ")"
        )


# This exists for backward compatibility
class LinearPreconditioner(AbstractLinearPreconditioner, mutable=True):
    lhs_constructor: LHSConstructorT
    """Constructor of the LHS of the linear system starting from the variational state."""

    def __init__(
        self,
        lhs_constructor: LHSConstructorT,
        solver: SolverT,
        *,
        solver_restart: bool = False,
    ):
        self._lhs_constructor = lhs_constructor
        self.solver = solver
        self.solver_restart = solver_restart

    def lhs_constructor(self, vstate: VariationalState, step: Scalar | None = None):
        """
        This method does things
        """
        return self._lhs_constructor(vstate, step)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"\n\tlhs_constructor = {self._lhs_constructor}, "
            + f"\n\tsolver          = {self.solver}, "
            + f"\n\tsolver_restart  = {self.solver_restart},"
            + ")"
        )
