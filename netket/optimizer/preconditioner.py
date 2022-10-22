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

from typing import Callable, Optional, Any

import abc
from dataclasses import dataclass

from netket.utils.types import PyTree, Scalar
from netket.vqs import VariationalState

from .linear_operator import LinearOperator, SolverT

# Generic signature of a preconditioner function/object

PreconditionerT = Callable[[VariationalState, PyTree, Optional[Scalar]], PyTree]
"""Signature for Gradient preconditioners supported by NetKet drivers."""

LHSConstructorT = Callable[[VariationalState, Optional[Scalar]], LinearOperator]
"""Signature for the constructor of a LinerOperator"""


def identity_preconditioner(
    vstate: VariationalState, gradient: PyTree, step: Scalar = 0
):
    return gradient


@dataclass
class AbstractLinearPreconditioner:
    """Linear Preconditioner for the gradient. Needs a function to construct the LHS of
    the Linear System and a solver to solve the linear system.
    """

    solver: SolverT
    """Function used to solve the linear system."""

    solver_restart: bool = False
    """If False uses the last solution of the linear system as a starting point for the solution
    of the next."""

    x0: Optional[PyTree] = None
    """Solution of the last linear system solved."""

    info: Any = None
    """Additional information returned by the solver when solving the last linear system."""

    _lhs: LinearOperator = None
    """LHS of the last linear system solved."""

    def __init__(self, solver, *, solver_restart=False):
        self.solver = solver
        self.solver_restart = solver_restart

    def __call__(
        self, vstate: VariationalState, gradient: PyTree, step: Optional[Scalar] = None
    ) -> PyTree:

        self._lhs = self.lhs_constructor(vstate, step)

        x0 = self.x0 if self.solver_restart else None
        self.x0, self.info = self._lhs.solve(self.solver, gradient, x0=x0)

        return self.x0

    @abc.abstractmethod
    def lhs_constructor(self, vstate: VariationalState, step: Optional[Scalar] = None):
        """
        This method does things
        """

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"\n\tsolver          = {self.solver}, "
            + f"\n\tsolver_restart  = {self.solver_restart},"
            + ")"
        )


# This exists for backward compatibility
@dataclass
class LinearPreconditioner(AbstractLinearPreconditioner):
    lhs_constructor: LHSConstructorT
    """Constructor of the LHS of the linear system starting from the variational state."""

    def __init__(self, lhs_constructor, solver, *, solver_restart=False):
        self._lhs_constructor = lhs_constructor
        self.solver = solver
        self.solver_restart = solver_restart

    def lhs_constructor(self, vstate: VariationalState, step: Optional[Scalar] = None):
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
