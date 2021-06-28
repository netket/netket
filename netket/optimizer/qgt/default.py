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
from functools import partial

import jax

import netket.jax as nkjax

from .qgt_jacobian_dense import QGTJacobianDense
from .qgt_jacobian_pytree import QGTJacobianPyTree
from .qgt_onthefly import QGTOnTheFly

from ..solver import cholesky, svd, LU, solve

solvers = [cholesky, svd, LU, solve]


def _is_dense_solver(solver: Any) -> bool:
    """
    Returns true if the solver is one of our known dense solvers
    """
    if isinstance(solver, partial):
        solver = solver.func

    if solver in solvers:
        return True

    return False


def default_qgt_matrix(variational_state, solver=False, **kwargs):

    n_param_leaves = len(jax.tree_leaves(variational_state.parameters))
    n_params = variational_state.n_parameters

    # those require dense matrix that is known to be faster for this qgt
    if _is_dense_solver(solver):
        return partial(QGTJacobianDense, **kwargs)

    # arbitrary heuristic: if the network's parameters has many leaves
    # (an rbm has 3) then JacobianDense might be faster
    # the numbers chosen below are rather arbitrary and should be tuned.
    if n_param_leaves > 6 and n_params > 800:
        if nkjax.tree_ishomogeneous(variational_state.variables):
            return partial(QGTJacobianDense, **kwargs)
        else:
            return partial(QGTJacobianPyTree, **kwargs)
    else:
        return partial(QGTOnTheFly, **kwargs)


class QGTAuto:
    """
    Automatically select the 'best' Quantum Geometric Tensor
    computing format acoording to some rather untested heuristic.

    Args:
        variational_state: The variational State
        kwargs: are passed on to the QGT constructor.
    """

    _last_vstate = None
    """Cached last variational state to skip logic to decide what type of
    QGT to chose.
    """

    _last_matrix = None
    """
    Cached last QGT. Used when vstate == _last_vstate
    """

    _kwargs = {}
    """
    Kwargs passed at construction. Used when constructing a QGT.
    """

    def __init__(self, solver=None, **kwargs):
        self._solver = solver

        self._kwargs = kwargs

    def __call__(self, variational_state, *args, **kwargs):
        if self._last_vstate != variational_state:
            self._last_vstate = variational_state

            self._last_matrix = default_qgt_matrix(
                variational_state, solver=self._solver, **self._kwargs, **kwargs
            )

        return self._last_matrix(variational_state, *args, **kwargs)

    def __repr__(self):
        return "QGTAuto()"
