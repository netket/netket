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

from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
import flax
from jax import numpy as jnp
from flax import struct

from plum import dispatch

from netket.utils.types import PyTree, Array

from .base import SR

from .sr_onthefly_logic import mat_vec as mat_vec_onthefly, tree_cast
from .s_onthefly_mat import SROnTheFly, SMatrixOnTheFly


@struct.dataclass
class SROnTheFlyIterative(SROnTheFly):
    """
    Base class holding the parameters for the iterative solution of the
    SR system x = ⟨S⟩⁻¹⟨F⟩, where S is a lazy linear operator

    Tolerances are applied according to the formula
    :code:`norm(residual) <= max(tol*norm(b), atol)`
    """

    tol: float = 1.0e-5
    """Relative tolerance for convergences."""

    atol: float = 0.0
    """Absolutes tolerance for convergences."""

    maxiter: int = None
    """Maximum number of iterations. Iteration will stop after maxiter steps even 
    if the specified tolerance has not been achieved.
    """

    M: Optional[Union[Callable, Array]] = None
    """Preconditioner for A. The preconditioner should approximate the inverse of A. 
    Effective preconditioning dramatically improves the rate of convergence, which implies 
    that fewer iterations are needed to reach a given error tolerance.
    """

    def create(self, vstate, **kwargs) -> "LazySMatrixIterative":
        """
        Construct the Lazy representation of the S corresponding to this SR type.

        Args:
            vstate: The Variational State
        """
        return LazySMatrixIterative(
            apply_fun=vstate._apply_fun,
            params=vstate.parameters,
            samples=vstate.samples,
            model_state=vstate.model_state,
            sr=self,
        )


@struct.dataclass
class SRLazyCG(SROnTheFlyIterative):
    """
    Computes x = ⟨S⟩⁻¹⟨F⟩ by using an iterative conjugate gradient method.

    See `Jax docs <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html#jax.scipy.sparse.linalg.cg>`_
    for more informations.
    """

    ...

    def solve_fun(self):
        return partial(
            jax.scipy.sparse.linalg.cg,
            tol=self.tol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=self.M,
        )


@struct.dataclass
class SRLazyGMRES(SROnTheFlyIterative):
    """
    Computes x = ⟨S⟩⁻¹⟨F⟩ by using an iterative GMRES method.

    See `Jax docs <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html#jax.scipy.sparse.linalg.gmres>`_
    for more informations.
    """

    restart: int = 20
    """Size of the Krylov subspace (“number of iterations”) built between restarts. 
    GMRES works by approximating the true solution x as its projection into a Krylov 
    space of this dimension - this parameter therefore bounds the maximum accuracy 
    achievable from any guess solution. Larger values increase both number of iterations 
    and iteration cost, but may be necessary for convergence. The algorithm terminates early 
    if convergence is achieved before the full subspace is built. 
    Default is 20
    """

    solve_method: str = struct.field(pytree_node=False, default="batched")
    """(‘incremental’ or ‘batched’) – The ‘incremental’ solve method builds a QR 
    decomposition for the Krylov subspace incrementally during the GMRES process 
    using Givens rotations. This improves numerical stability and gives a free estimate 
    of the residual norm that allows for early termination within a single “restart”. In 
    contrast, the ‘batched’ solve method solves the least squares problem from scratch at 
    the end of each GMRES iteration. It does not allow for early termination, but has much 
    less overhead on GPUs.
    """

    def solve_fun(self):
        return partial(
            jax.scipy.sparse.linalg.gmres,
            tol=self.tol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=self.M,
            restart=self.restart,
            solve_method=self.solve_method,
        )


@struct.dataclass
class LazySMatrixIterative(SMatrixOnTheFly):
    """
    Lazy representation of an S Matrix behving like a linear operator.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    x0: Optional[PyTree] = None
    """Optional initial guess for the iterative solution."""

    def solve(self, y: PyTree, x0: Optional[PyTree] = None, **kwargs) -> PyTree:
        """
        Solve the linear system x=⟨S⟩⁻¹⟨y⟩ with the chosen iterataive solver.

        Args:
            y: the vector y in the system above.
            x0: optional initial guess for the solution.

        Returns:
            x: the PyTree solving the system.
            info: optional additional informations provided by the solver. Might be
                None if there are no additional informations provided.
        """
        if x0 is None:
            x0 = self.x0

        out, info = apply_onthefly(
            self,
            y,
            x0,
        )

        return out, info


@jax.jit
def apply_onthefly(
    S: LazySMatrixIterative, grad: PyTree, x0: Optional[PyTree]
) -> PyTree:
    # Preapply the model state so that when computing gradient we only
    # get gradient of parameeters
    def fun(W, σ):
        return S.apply_fun({"params": W, **S.model_state}, σ)

    grad = tree_cast(grad, S.params)
    # we could cache this...
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, grad)

    _mat_vec = partial(
        mat_vec_onthefly,
        forward_fn=fun,
        params=S.params,
        samples=S.samples,
        diag_shift=S.sr.diag_shift,
        centered=S.sr.centered,
    )
    solve_fun = S.sr.solve_fun()
    out, info = solve_fun(_mat_vec, grad, x0=x0)
    return out, info
