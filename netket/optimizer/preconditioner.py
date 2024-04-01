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
from textwrap import dedent

from netket.utils.types import PyTree, Scalar
from netket.utils import warn_deprecation, timing
from netket.vqs import VariationalState

from .linear_operator import LinearOperator, SolverT

# Generic signature of a preconditioner function/object

PreconditionerT = Callable[[VariationalState, PyTree, Optional[Scalar]], PyTree]
"""Signature for Gradient preconditioners supported by NetKet drivers."""

LHSConstructorT = Callable[[VariationalState, Optional[Scalar]], LinearOperator]
"""Signature for the constructor of a LinerOperator"""


def identity_preconditioner(
    vstate: VariationalState, gradient: PyTree, step: Optional[Scalar] = 0
) -> PyTree:
    return gradient


@dataclass
class AbstractLinearPreconditioner:
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

    @timing.timed
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
@dataclass
class LinearPreconditioner(AbstractLinearPreconditioner):
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


class DeprecatedPreconditionerSignature:
    """
    Ignores the step argument for old-syntax preconditioners.
    """

    def __init__(self, fun):
        self.preconditioner = fun
        warn_deprecation(
            dedent(
                """

            Preconditioners that only accept two arguments are deprecated since
            version 3.7 and will no longer be supported in a future version.

            Preconditioners should now accept 3 arguments, where the first two
            are a Variational State and the gradient, while the last one is an
            optional scalar value representing the current step along the
            optimisation and which can be used to update some hyperparamter
            along the optimisation.

            To silence this deprecation warning, either modify your preconditioner
            or do the following:

            >>> driver.preconditioner = lambda state, grad, step=None: fun(state, grad)

            """
            )
        )

    def __call__(self, vstate: VariationalState, gradient: PyTree, step: Scalar = None):
        return self.preconditioner(vstate, gradient)
