# Copyright 2022 The NetKet Authors - All rights reserved.
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

from netket.utils.types import PyTree
from netket.utils import mpi
import netket.jax as nkjax

from ..linear_operator import LinearOperator, Uninitialized

from netket.optimizer.qgt.qgt_jacobian_dense_logic import (
    prepare_centered_oks,
    vec_to_real,
)
from netket.optimizer.qgt.qgt_jacobian_common import choose_jacobian_mode


def NTKJacobianDense(
    vstate=None,
    *,
    mode: str = None,
    holomorphic: bool = None,
    rescale_shift=False,
    **kwargs,
) -> "QGTJacobianDenseT":
    """
    Semi-lazy representation of an S Matrix where the Jacobian O_k is precomputed
    and stored as a dense matrix.

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
        rescale_shift: If True rescales the diagonal shift
    """

    if vstate is None:
        return partial(
            NTKJacobianDense,
            mode=mode,
            holomorphic=holomorphic,
            rescale_shift=rescale_shift,
            **kwargs,
        )

    # TODO: Find a better way to handle this case
    from netket.vqs import ExactState

    if isinstance(vstate, ExactState):
        raise TypeError("Only QGTJacobianPyTree works with ExactState.")

    if mode is None:
        mode = choose_jacobian_mode(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            vstate.samples,
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
        vstate.samples.reshape(-1, vstate.samples.shape[-1]),
        vstate.model_state,
        mode,
        rescale_shift,
        chunk_size,
    )

    return NTKJacobianDenseT(O=O, scale=scale, mode=mode, **kwargs)


@struct.dataclass
class NTKJacobianDenseT(LinearOperator):
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

    def __matmul__(self, vec: Union[PyTree, jnp.ndarray]) -> Union[PyTree, jnp.ndarray]:
        return _matmul(self, vec)

    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None) -> PyTree:
        return _solve(self, solve_fun, y, x0=x0)

    def project_to(self, vec: Union[PyTree, jnp.ndarray], pars):
        return _project_to(self, vec, pars)

    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        return _to_dense(self)

    def __repr__(self):
        return (
            f"NTKJacobianDense(diag_shift={self.diag_shift}, "
            f"scale={self.scale}, mode={self.mode})"
        )


########################################################################################
#####                                  QGT Logic                                   #####
########################################################################################

@jax.jit
def _project_to(self, vec, pars):
    #if self.mode != "holomorphic" and not self._in_solve:
    #    vec, reassemble = vec_to_real(vec)

    w = self.O.T.conj() @ vec

    #if reassemble is not None:
    #    w = reassemble(w)

    _, unravel = nkjax.tree_ravel(pars)
    return unravel(w)


@jax.jit
def _matmul(
    self: NTKJacobianDenseT, vec: Union[PyTree, jnp.ndarray]
) -> Union[PyTree, jnp.ndarray]:

    # Real-imaginary split RHS in R→R and R→C modes
    reassemble = None
    if self.mode != "holomorphic" and not self._in_solve:
        vec, reassemble = vec_to_real(vec)

    if self.scale is not None:
        vec = vec * self.scale

    w = self.O.T.conj() @ vec
    w̄ = mpi.mpi_sum_jax(w)[0]
    result = self.O @ w̄ + self.diag_shift * vec

    if self.scale is not None:
        result = result * self.scale

    if reassemble is not None:
        result = reassemble(result)

    return result


@jax.jit
def _solve(
    self: NTKJacobianDenseT, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None
) -> PyTree:

    if self.mode != "holomorphic":
        y, reassemble = vec_to_real(y)

    if x0 is not None:
        x0, _ = nkjax.tree_ravel(x0)
        if self.mode != "holomorphic":
            x0, _ = vec_to_real(x0)

        if self.scale is not None:
            x0 = x0 * self.scale

    if self.scale is not None:
        y = y / self.scale

    # to pass the object LinearOperator itself down
    # but avoid rescaling, we pass down an object with
    # scale = None
    unscaled_self = self.replace(scale=None, _in_solve=True)

    out, info = solve_fun(unscaled_self, y, x0=x0)

    if self.scale is not None:
        out = out / self.scale

    if self.mode != "holomorphic":
        out = reassemble(out)

    return out, info


@jax.jit
def _to_dense(self: NTKJacobianDenseT) -> jnp.ndarray:
    if self.scale is None:
        O = self.O
        diag = jnp.eye(self.O.shape[0])
    else:
        O = self.O * self.scale[jnp.newaxis, :]
        diag = jnp.diag(self.scale**2)

    return O @ O.T.conj() + self.diag_shift * diag
