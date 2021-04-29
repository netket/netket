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

from netket.utils.types import PyTree, Array
from netket.utils import rename_class
import netket.jax as nkjax

from .s_jacobian_mat import JacobianSMatrix, gradients

from .base import SR


@struct.dataclass
class SRJacobian(SR):
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

    mode: str = struct.field(pytree_node=False, default="invalid")
    """Differentiation mode to precompute Jacobian
    * "holomorphic": C->C holomorphic function
        `grad` is called on the full network output with `holomorphic=True`
    * "R2R": real-valued wave function with real parameters
        `grad` is called on the real part of the network output with `holomorphic=False`
    * "R2C": complex-valued wave function with real parameters
        the real and imaginary parts of the network output are treated as independent 
        R->R functions and `grad` is called separately on them with `holomorphic=False`
    * the default value "invalid" will trigger an error
    """

    rescale_shift: bool = struct.field(pytree_node=False, default=False)
    """Whether scale-invariant regularisation should be used"""

    def __post_init__(self):
        if self.mode not in {"R2R", "R2C", "holomorphic"}:
            raise NotImplementedError(
                'Differentiation mode must be one of "R2R", "R2C", "holomorphic", got "{}"'.format(
                    self.mode
                )
            )

    def create(self, vstate, **kwargs) -> "LazySMatrixIterative":
        """
        Construct the Lazy representation of the S corresponding to this SR type.

        Args:
            vstate: The Variational State
        """
        O, scale = gradients(
            vstate._apply_fun,
            vstate.parameters,
            vstate.samples,
            vstate.model_state,
            self.mode,
            self.rescale_shift,
        )
        return JacobianSMatrix(sr=self, O=O, scale=scale)


@struct.dataclass
class SRJacobianCG(SRJacobian):
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
class SRJacobianGMRES(SRJacobian):
    """
    Computes x = ⟨S⟩⁻¹⟨F⟩ by using an iterative GMRES method.

    See `Jax docs <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html#jax.scipy.sparse.linalg.gmres>`_
    for more informations.
    """

    restart: int = struct.field(pytree_node=False, default=20)
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
