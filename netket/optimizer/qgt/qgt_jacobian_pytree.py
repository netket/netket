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

from .qgt_jacobian_pytree_logic import mat_vec, prepare_doks


def QGTJacobianPyTree(vstate, *, mode, rescale_shift=True, **kwargs) -> "QGTJacobianPyTreeT":
    O, scale = prepare_doks(vstate._apply_fun,
        vstate.parameters,
        vstate.samples,
        vstate.model_state,
        mode,
        rescale_shift,
    )

    _, unravel = nkjax.tree_ravel(vstate.parameters)

    return QGTJacobianPyTreeT(O=O, scale=scale, unravel=unravel, **kwargs)


@struct.dataclass
class QGTJacobianPyTreeT(LinearOperator):
    """
    Semi-lazy representation of an S Matrix behaving like a linear operator.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    O: PyTree = Uninitialized
    """Centred gradients ΔO_ij = O_ij - <O_j> of the neural network, where
    O_ij = ∂log ψ(σ_i)/∂p_j, for all samples σ_i at given values of the parameters p_j
    Divided through with sqrt(#samples) to normalise S matrix
    If scale is not None, O_ij for is normalised to unit norm for each parameter j
    """

    scale: Optional[PyTree] = None
    """If not None, contains 2-norm of each column of the gradient matrix,
    i.e., the sqrt of the diagonal elements of the S matrix
    """

    unravel: Callable = struct.field(
        pytree_node=False, default=Uninitialized
    )
    """Function to unravel input vectors in __matmul__"""

    @jax.jit
    def __matmul__(self, vec: Union[PyTree, jnp.ndarray]) -> Union[PyTree, jnp.ndarray]:
        if hasattr(vec, "ndim"):
            vec = self.unravel(vec)
            ravel = True
        else:
            ravel = False

        if self.scale is not None:
            vec = jax.tree_multimap(jnp.multiply, vec, self.scale)

        result = mat_vec(vec, self.O, self.diag_shift)

        if self.scale is not None:
            result = jax.tree_multimap(jnp.multiply, result, self.scale)

        if ravel:
            result, _ = nkjax.tree_ravel(result)

        return result        

    @jax.jit
    def _unscaled_matmul(self, vec: PyTree) -> PyTree:
        return mat_vec(vec, self.O, self.diag_shift)

    
    @partial(jax.jit, static_argnums=1)
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
        if self.scale is not None:
            y = jax.tree_multimap(jnp.divide, y, self.scale)
            if x0 is not None:
                x0 = jax.tree_multimap(jnp.multiply, x0, self.scale)

        out, info = solve_fun(self._unscaled_matmul, y, x0=x0)

        if self.scale is not None:
            out = jax.tree_multimap(jnp.divide, out, self.scale)

        return out, info

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        raise NotImplementedError
