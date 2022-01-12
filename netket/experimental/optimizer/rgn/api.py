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

from functools import partial
from collections import namedtuple

from netket.utils.numbers import is_scalar
from netket.vqs import VariationalState
from netket.utils.types import PyTree
from netket.operator import AbstractOperator

import jax

from .hessian_plus_qgt_pytree import Hessian_Plus_QGT_PyTree
from netket.optimizer.preconditioner import LinearPreconditioner

Preconditioner = namedtuple("Preconditioner", ["object", "solver"])

default_iterative = "cg"


class RGN(LinearPreconditioner):
    def __call__(self, gradient: PyTree) -> PyTree:

        self._lhs = self.lhs_constructor()

        x0 = self.x0 if self.solver_restart else None
        self.x0, self.info = self._lhs.solve(self.solver, gradient, x0=x0)

        return self.x0


def _RGN(
    solver=None,
    solver_restart: bool = False,
    *args,
    **kwargs,
):
    """
    Construct the structure holding the parameters for using the
    Stochastic Reconfiguration/Natural gradient method.

    Depending on the arguments, an implementation is chosen. For
    details on all possible kwargs check the specific SR implementations
    in the documentation.

    You can also construct one of those structures directly.

    Args:
        diag_shift: Diagonal shift added to the S matrix
        method: (cg, gmres) The specific method.
        iterative: Whever to use an iterative method or not.
        jacobian: Differentiation mode to precompute gradients
                  can be "holomorphic", "R2R", "R2C",
                         None (if they shouldn't be precomputed)
        rescale_shift: Whether to rescale the diagonal offsets in SR according
                       to diagonal entries (only with precomputed gradients)

    Returns:
        The SR parameter structure.
    """

    if solver is None:
        solver = jax.scipy.sparse.linalg.cg

    return RGN(
        partial(Hessian_Plus_QGT_PyTree, *args, **kwargs),
        solver=solver,
        solver_restart=solver_restart,
    )
