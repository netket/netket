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

from netket.utils import wraps_legacy
from netket.legacy.optimizer import SR as SR_legacy
from netket.legacy.machine import AbstractMachine

from netket.utils.numbers import is_scalar

import jax

from ..qgt import QGTAuto
from ..preconditioner import LinearPreconditioner

Preconditioner = namedtuple("Preconditioner", ["object", "solver"])

default_iterative = "cg"
# default_direct = "eigen"


@wraps_legacy(SR_legacy, "machine", AbstractMachine)
def build_SR(*args, solver_restart: bool = False, **kwargs):
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

    #  try to understand if this is the old API or new
    # API

    old_api = False
    # new_api = False

    if "matrix" in kwargs:
        # new syntax
        return _SR(*args, **kwargs)

    legacy_kwargs = ["iterative", "method"]
    legacy_solver_kwargs = ["tol", "atol", "maxiter", "M", "restart", "solve_method"]
    for key in legacy_kwargs + legacy_solver_kwargs:
        if key in kwargs:
            old_api = True
            break

    if len(args) > 0:
        if is_scalar(args[0]):  # it's diag_shift
            old_api = True
        # else:
        #    new_api = True

        if len(args) > 1:
            if isinstance(args[1], str):
                old_api = True
        #     else:
        #        new_api = True

    if old_api:
        for (i, arg) in enumerate(args):
            if i == 0:
                # diag shift
                kwargs["diag_shift"] = arg
            elif i == 1:
                kwargs["method"] = arg
            else:
                raise TypeError(
                    "SR takes at most 2 positional arguments but len(args) where provided"
                )

        args = tuple()

        solver = None
        if "iterative" in kwargs:
            kwargs.pop("iterative")
        if "method" in kwargs:
            legacy_solvers = {
                "cg": jax.scipy.sparse.linalg.cg,
                "gmres": jax.scipy.sparse.linalg.gmres,
            }
            if kwargs["method"] not in legacy_solvers:
                raise ValueError(
                    "The old API only supports cg and gmres solvers. "
                    "Migrate to the new API and use any solver from"
                    "jax.scipy.sparse.linalg"
                )
            solver = legacy_solvers[kwargs["method"]]
            kwargs.pop("method")
        else:
            solver = jax.scipy.sparse.linalg.cg

        solver_keys = {}
        has_solver_kw = False
        for key in legacy_solver_kwargs:
            if key in kwargs:
                solver_keys[key] = kwargs[key]
                has_solver_kw = True

        if has_solver_kw:
            solver = partial(solver, **solver_keys)

        kwargs["solver"] = solver

    return _SR(*args, solver_restart=solver_restart, **kwargs)


class SR(LinearPreconditioner):
    pass


# This will become the future implementation once legacy and semi-legacy
# bejaviour is removed
def _SR(
    qgt=None,
    solver=None,
    *,
    diag_shift: float = 0.01,
    solver_restart: bool = False,
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
        qgt: The Quantum Geomtric Tensor type to use.
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

    if qgt is None:
        qgt = QGTAuto(solver)

    return SR(
        partial(qgt, diag_shift=diag_shift, **kwargs),
        solver=solver,
        solver_restart=solver_restart,
    )
