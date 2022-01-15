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

import jax

from netket.utils import deprecated, warn_deprecation

from .api import build_SR as SR
from .. import qgt


@deprecated(
    """
    SRLazyCG(diag_shift, max_iters,...) has been deprecated and will be
    soon removed as soon as the beta is finished.
    We have split the solver (CG) part from the matrix representation (SRLazy).
    You should create an sr object passing the name of the Quantum Geometric
    Tensor storage format (in this case OnTheFly) and the jax solver.
    You can wrap arguments from the solver with `functools.partial`

    >>> solver = partial(jax.scipy.sparse.cg, max_iter=, ...)
    >>> nk.optimizer.SR(nk.optimizer.qgt.QGTOnTheFly, solver=solver, diag_shift=0.01)
    """
)
def SRLazyCG(diag_shift: float = 0.01, centered: bool = None, **kwargs):

    if centered is not None:
        warn_deprecation(
            "The argument `centered` is deprecated. The implementation now always behaves as if centered=False."
        )

    return SR(
        qgt.QGTOnTheFly,
        solver=partial(jax.scipy.sparse.linalg.cg, **kwargs),
        diag_shift=diag_shift,
        **kwargs,
    )


@deprecated(
    """
    SRLazyGMRES(diag_shift, max_iters,...) has been deprecated and will be
    soon removed as soon as the beta is finished.
    We have split the solver (CG) part from the matrix representation (SRLazy).
    You should create an sr object passing the name of the Quantum Geometric
    Tensor storage format (in this case OnTheFly) and the jax solver.
    You can wrap arguments from the solver with `functools.partial`

    >>> solver = partial(jax.scipy.sparse.cg, max_iter=, ...)
    >>> nk.optimizer.SR(nk.optimizer.qgt.QGTOnTheFly, solver=solver, diag_shift=0.01)
    """
)
def SRLazyGMRES(diag_shift: float = 0.01, centered: bool = None, **kwargs):

    if centered is not None:
        warn_deprecation(
            "The argument `centered` is deprecated. The implementation now always behaves as if centered=False."
        )

    return SR(
        qgt.QGTOnTheFly,
        solver=partial(jax.scipy.sparse.linalg.gmres, **kwargs),
        diag_shift=diag_shift,
        **kwargs,
    )
