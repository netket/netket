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
from jax import numpy as jnp
from flax import struct

from netket.utils.types import PyTree
from netket.utils import n_nodes
from netket.stats import sum_inplace
import netket.jax as nkjax

from ..linear_operator import LinearOperator, Uninitialized

from .qgt_jacobian_pytree_logic import mat_vec


def QGTJacobianPyTree(vstate, *, mode, rescale_shift=True) -> "QGTJacobianPyTreeT":
    O, scale = 0, 0  # compute gradient

    return QGTJacobianT(O=O, scale=rescale_shift)


@struct.dataclass
class QGTJacobianPyTreeT(LinearOperator):
    """
    Semi-lazy representation of an S Matrix behaving like a linear operator.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    O: jnp.ndarray = Uninitialized
    """Gradients O_ij = ∂log ψ(σ_i)/∂p_j of the neural network 
    for all samples σ_i at given values of the parameters p_j
    Average <O_j> subtracted for each parameter
    Divided through with sqrt(#samples) to normalise S matrix
    If scale is not None, columns normalised to unit norm
    """

    scale: Optional[jnp.ndarray] = None
    """If not None, contains 2-norm of each column of the gradient matrix,
    i.e., the sqrt of the diagonal elements of the S matrix
    """

    @jax.jit
    def __matmul__(self, vec: Union[PyTree, jnp.ndarray]) -> Union[PyTree, jnp.ndarray]:
        mat_vec(vec, self.O, self.scale)

    @jax.jit
    def _unscaled_matmul(self, vec: jnp.ndarray) -> jnp.ndarray:
        return (
            sum_inplace(((self.O @ vec).T.conj() @ self.O).T.conj())
            + self.diag_shift * vec
        )

    @jax.jit
    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None) -> PyTree:
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

        # Ravel input PyTrees, record unravelling function too
        grad, unravel = nkjax.tree_ravel(y)

        if x0 is not None:
            x0, _ = nkjax.tree_ravel(x0)
            if self.scale is not None:
                x0 = x0 * self.scale

        if self.scale is not None:
            grad = grad / self.scale

        solve_fun = self.sr.solve_fun()
        out, info = solve_fun(self._unscaled_matmul, grad, x0=x0)

        if self.scale is not None:
            out = out / self.scale

        return unravel(out), info

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        if scale is None:
            O = self.O
            diag = jnp.eye(self.O.shape[1])
        else:
            O = self.O * self.scale[jnp.newaxis, :]
            diag = jnp.diag(self.scale ** 2)

        return sum_inplace(O.T.conj() @ O) + self.diag_shift * diag


def _grad_vmap_minus_mean(
    fun: Callable, params: jnp.ndarray, samples: jnp.ndarray, holomorphic: bool
):
    """Calculates the gradient of a neural network for a number of samples
    efficiently using vmap(grad),
    and subtracts their mean for each parameter, i.e., each column
    """
    grads = jax.vmap(
        jax.grad(fun, holomorphic=holomorphic), in_axes=(None, 0), out_axes=0
    )(params, samples)
    return grads - sum_inplace(grads.sum(axis=0, keepdims=True)) / (
        grads.shape[0] * n_nodes
    )
