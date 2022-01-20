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

from typing import Optional, Union
from functools import partial

import jax
from jax import numpy as jnp
from flax import struct

from netket.utils.types import PyTree, Array
from netket.utils import mpi
import netket.jax as nkjax

from netket.optimizer.linear_operator import LinearOperator, Uninitialized
from .model_and_operator_statistics import mat_vec

from netket.nn import split_array_mpi


@struct.dataclass
class Hessian_Plus_QGT_PyTree(LinearOperator):
    """
    Semi-lazy representation of an S Matrix behaving like a linear operator.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    jac: PyTree = Uninitialized
    """Centred gradients ΔO_ij = O_ij - <O_j> of the neural network, where
    O_ij = ∂log ψ(σ_i)/∂p_j, for all samples σ_i at given values of the parameters p_j
    Divided through with sqrt(#samples) to normalise S matrix
    If scale is not None, O_ij for is normalised to unit norm for each parameter j
    """

    jac_mean: PyTree = Uninitialized
    """Average of the jacobian needed for the matvec"""

    rhes: PyTree = Uninitialized
    """Right hand side of the hessian to compute the first term in the RGN expansion"""

    grad: PyTree = Uninitialized
    """gradient of the energy with respect to the parameters"""

    energy: float = Uninitialized
    """energy of the variational state"""

    eps: float = Uninitialized
    """scale of the qgt contribution relative to the Hessian"""

    diag_shift: float = Uninitialized
    """diagonal shift applied to the full matrix"""

    params: PyTree = Uninitialized
    """Parameters of the network. Its only purpose is to represent its own shape when scale is None"""

    mode: str = struct.field(pytree_node=False, default=Uninitialized)
    """Differentiation mode:
        - "real": for real-valued R->R and C->R ansatze, splits the complex inputs
                  into real and imaginary part.
        - "complex": for complex-valued R->C and C->C ansatze, splits the complex
                  inputs and outputs into real and imaginary part
        - "holomorphic": for any ansatze. Does not split complex values.
    """

    _in_solve: bool = struct.field(pytree_node=False, default=False)
    """Internal flag used to signal that we are inside the _solve method and matmul should
    not take apart into real and complex parts the other vector"""

    def __matmul__(self, vec: Union[PyTree, Array]) -> Union[PyTree, Array]:
        return _matmul(self, vec)

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
        return _solve(self, solve_fun, y, x0=x0)

    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
            In R→R and R→C modes, real and imaginary parts of parameters get own rows/columns
        """
        return _to_dense(self)

    def __repr__(self):
        return (
            f"RGNPyTree(diag_shift={self.diag_shift}, "
            f"scale={self.scale}, mode={self.mode})"
        )


@jax.jit
def _matmul(
    self: Hessian_Plus_QGT_PyTree, vec: Union[PyTree, Array]
) -> Union[PyTree, Array]:
    # Turn vector RHS into PyTree
    if hasattr(vec, "ndim"):
        _, unravel = nkjax.tree_ravel(self.params)
        vec = unravel(vec)
        ravel = True
    else:
        ravel = False

    # Real-imaginary split RHS in R→R and R→C modes
    reassemble = None
    if self.mode != "holomorphic" and not self._in_solve:
        vec, reassemble = nkjax.tree_to_real(vec)

    result = mat_vec(
        vec, self.jac, self.rhes, self.jac_mean, self.eps, self.energy, self.diag_shift
    )

    # Reassemble real-imaginary split as needed
    if reassemble is not None:
        result = reassemble(result)

    # Ravel PyTree back into vector as needed
    if ravel:
        result, _ = nkjax.tree_ravel(result)

    return result


@jax.jit
def _solve(
    self: Hessian_Plus_QGT_PyTree, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None
) -> PyTree:
    # Real-imaginary split RHS in R→R and R→C modes
    if self.mode != "holomorphic":
        y, reassemble = nkjax.tree_to_real(y)

    out, info = solve_fun(self, y, x0=x0)

    # Reassemble real-imaginary split as needed
    if self.mode != "holomorphic":
        out = reassemble(out)

    return out, info


@jax.jit
def _to_dense(self: Hessian_Plus_QGT_PyTree) -> jnp.ndarray:

    raise NotImplementedError
