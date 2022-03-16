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

from ..linear_operator import LinearOperator, Uninitialized

from netket.optimizer.qgt.common import check_valid_vector_type
from netket.optimizer.qgt.qgt_jacobian_pytree_logic import prepare_centered_oks
from netket.optimizer.qgt.qgt_jacobian_common import choose_jacobian_mode

from netket.nn import split_array_mpi


def NTKJacobianPyTree(
    vstate=None,
    *,
    mode: str = None,
    holomorphic: bool = None,
    rescale_shift=False,
    **kwargs,
) -> "NTKJacobianPyTreeT":
    """
    Semi-lazy representation of an S Matrix where the Jacobian O_k is precomputed
    and stored as a PyTree.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.

    Args:
        vstate: The variational state
        mode: "real", "complex" or "holomorphic": specifies the implementation
              used to compute the jacobian. "real" discards the imaginary part
              of the output of the model. "complex" splits the real and imaginary
              part of the parameters and output. It works also for non holomorphic
              models. holomorphic works for any function assuming it's holomorphic
              or real valued.
        holomorphic: a flag to indicate that the function is holomorphic.
        rescale_shift: If True rescales the diagonal shift.
    """
    if vstate is None:
        return partial(
            NTKJacobianPyTree,
            mode=mode,
            holomorphic=holomorphic,
            rescale_shift=rescale_shift,
            **kwargs,
        )

    # TODO: Find a better way to handle this case
    from netket.vqs import ExactState

    if isinstance(vstate, ExactState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    else:
        samples = vstate.samples
        pdf = None

    # Choose sensible default mode
    if mode is None:
        mode = choose_jacobian_mode(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            samples,
            mode=mode,
            holomorphic=holomorphic,
        )
    elif holomorphic is not None:
        raise ValueError("Cannot specify both `mode` and `holomorphic`.")

    if hasattr(vstate, "chunk_size"):
        chunk_size = vstate.chunk_size
    else:
        chunk_size = None

    O, scale = prepare_centered_oks(
        vstate._apply_fun,
        vstate.parameters,
        samples.reshape(-1, samples.shape[-1]),
        vstate.model_state,
        mode,
        rescale_shift,
        pdf,
        chunk_size,
    )

    return NTKJacobianPyTreeT(
        O=O, scale=scale, params=vstate.parameters, mode=mode, **kwargs
    )


@struct.dataclass
class NTKJacobianPyTreeT(LinearOperator):
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

    params: PyTree = Uninitialized
    """Parameters of the network. Its only purpose is to represent its own shape when scale is None"""

    mode: str = struct.field(pytree_node=False, default=Uninitialized)
    """Differentiation mode:
        - "real": for real-valued R->R and C->R ansatze, splits the complex inputs
                  into real and imaginary part.
        - "complex": for complex-valued R->C and C->C ansatze, splits the complex
                  inputs and outputs into real and imaginary part
        - "holomorphic": for any ansatze. Does not split complex values.
        - "auto": autoselect real or complex.
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

    def project_to(self, vec: Union[PyTree, jnp.ndarray], pars):
        return _project_to(self, vec, pars)

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
            f"NTKJacobianPyTree(diag_shift={self.diag_shift}, "
            f"scale={self.scale}, mode={self.mode})"
        )


# from netket.optimizer.qgt.qgt_jacobian_pytree_logic import _vjp, _jvp
from netket.jax import tree_cast, tree_conj, tree_axpy
from netket.utils.types import Array, Callable, PyTree, Scalar


def _jvp(oks: PyTree, v: PyTree) -> Array:
    """
    Compute the matrix-vector product between the pytree jacobian oks and the pytree vector v
    """
    td = lambda x, y: jnp.einsum("i...,...->i", x, y)
    # td = lambda x, y: jnp.tensordot(x, y, axes=y.ndim)
    return jax.tree_util.tree_reduce(jnp.add, jax.tree_multimap(td, oks, v))


def _vjp(oks: PyTree, w: Array) -> PyTree:
    """
    Compute the vector-matrix product between the vector w and the pytree jacobian oks
    """
    res = jax.tree_map(lambda j: jnp.tensordot(w, j, axes=1), oks)
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # MPI


def _mat_vec(v: PyTree, oks: PyTree) -> PyTree:
    """
    Compute ⟨O O†⟩v = ∑ₗ ⟨Oₖ Oₗᴴ⟩ vₗ
    """
    return _jvp(oks, tree_conj(_vjp(oks, v.conjugate())))


def mat_vec(v: PyTree, centered_oks: PyTree, diag_shift: Scalar) -> PyTree:
    """
    Compute (S + δ) v = 1/n ⟨ΔO† ΔO⟩v + δ v = ∑ₗ 1/n ⟨ΔOₖᴴΔOₗ⟩ vₗ + δ vₗ
    """
    return tree_axpy(diag_shift, v, _mat_vec(v, centered_oks))


@jax.jit
def _matmul(
    self: NTKJacobianPyTreeT, vec: Union[PyTree, Array]
) -> Union[PyTree, Array]:

    # Real-imaginary split RHS in R→R and R→C modes
    reassemble = None
    if self.mode != "holomorphic" and not self._in_solve:
        vec, reassemble = nkjax.tree_to_real(vec)

    # check_valid_vector_type(self.params, vec)

    if self.scale is not None:
        vec = jax.tree_multimap(jnp.multiply, vec, self.scale)

    result = mat_vec(vec, self.O, self.diag_shift)

    if self.scale is not None:
        result = jax.tree_multimap(jnp.multiply, result, self.scale)

    # Reassemble real-imaginary split as needed
    if reassemble is not None:
        result = reassemble(result)

    return result

@jax.jit
def _project_to(self, vec, pars):
    # Real-imaginary split RHS in R→R and R→C modes
    #reassemble = None
    #if self.mode != "holomorphic" and not self._in_solve:
    #    vec, reassemble = nkjax.tree_to_real(vec)

    w = tree_conj(_vjp(self.O, vec.conjugate()))

    #if reassemble is not None:
    #    w = reassemble(w)
    return w

@jax.jit
def _solve(
    self: NTKJacobianPyTreeT, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None
) -> PyTree:

    # Real-imaginary split RHS in R→R and R→C modes
    if self.mode != "holomorphic":
        y, reassemble = nkjax.tree_to_real(y)
        if x0 is not None:
            x0, _ = nkjax.tree_to_real(x0)

    # check_valid_vector_type(self.params, y)

    if self.scale is not None:
        y = jax.tree_multimap(jnp.divide, y, self.scale)
        if x0 is not None:
            x0 = jax.tree_multimap(jnp.multiply, x0, self.scale)

    # to pass the object LinearOperator itself down
    # but avoid rescaling, we pass down an object with
    # scale = None
    # mode=holomoprhic to disable splitting the complex part
    unscaled_self = self.replace(scale=None, _in_solve=True)

    out, info = solve_fun(unscaled_self, y, x0=x0)

    if self.scale is not None:
        out = jax.tree_multimap(jnp.divide, out, self.scale)

    # Reassemble real-imaginary split as needed
    if self.mode != "holomorphic":
        out = reassemble(out)

    return out, info


@jax.jit
def _to_dense(self: NTKJacobianPyTreeT) -> jnp.ndarray:
    O = jax.vmap(lambda l: nkjax.tree_ravel(l)[0])(self.O)

    if self.scale is None:
        diag = jnp.eye(O.shape[0])
    else:
        scale, _ = nkjax.tree_ravel(self.scale)
        O = O * scale[jnp.newaxis, :]
        diag = jnp.diag(scale**2)

    return O @ O.T.conj() + self.diag_shift * diag
