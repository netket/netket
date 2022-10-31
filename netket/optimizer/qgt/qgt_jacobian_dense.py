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

from netket.utils.types import PyTree
from netket.utils import mpi
import netket.jax as nkjax
from netket.nn import split_array_mpi

from ..linear_operator import LinearOperator, Uninitialized

from .common import check_valid_vector_type
from .qgt_jacobian_dense_logic import vec_to_real, mat_vec
from .qgt_jacobian_common import (
    choose_jacobian_mode,
    sanitize_diag_shift,
    to_shift_offset,
    rescale,
)


def QGTJacobianDense(
    vstate=None,
    *,
    mode: str = None,
    holomorphic: bool = None,
    diag_shift=None,
    diag_scale=None,
    rescale_shift=None,
    chunk_size=None,
    **kwargs,
) -> "QGTJacobianDenseT":
    """
    Semi-lazy representation of an S Matrix where the Jacobian O_k is precomputed
    and stored as a dense matrix.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.

    Numerical estimates of the QGT are usually ill-conditioned and require
    regularisation. The standard approach is to add a positive constant to the diagonal;
    alternatively, Becca and Sorella (2017) propose scaling this offset with the
    diagonal entry itself. NetKet allows using both in tandem:

    .. math::

        S_{ii} \\mapsto S_{ii} + \\epsilon_1 S_{ii} + \\epsilon_2;

    :math:`\\epsilon_{1,2}` are specified using `diag_scale` and `diag_shift`,
    respectively.

    Args:
        vstate: The variational state
        mode: "real", "complex" or "holomorphic": specifies the implementation
              used to compute the jacobian. "real" discards the imaginary part
              of the output of the model. "complex" splits the real and imaginary
              part of the parameters and output. It works also for non holomorphic
              models. holomorphic works for any function assuming it's holomorphic
              or real valued.
        holomorphic: a flag to indicate that the function is holomorphic.
        diag_scale: Fractional shift :math:`\\epsilon_1` added to diagonal entries (see above).
        diag_shift: Constant shift :math:`\\epsilon_2` added to diagonal entries (see above).
        chunk_size: If supplied, overrides the chunk size of the variational state
                    (useful for models where the backward pass requires more
                    memory than the forward pass).
    """
    if mode is not None and holomorphic is not None:
        raise ValueError("Cannot specify both `mode` and `holomorphic`.")
    if rescale_shift is not None and diag_scale is not None:
        raise ValueError("Cannot specify both `rescale_shift` and `diag_scale`.")

    if vstate is None:
        return partial(
            QGTJacobianDense,
            mode=mode,
            holomorphic=holomorphic,
            chunk_size=chunk_size,
            diag_shift=diag_shift,
            diag_scale=diag_scale,
            rescale_shift=rescale_shift,
            **kwargs,
        )

    diag_shift, diag_scale = sanitize_diag_shift(diag_shift, diag_scale, rescale_shift)

    # TODO: Find a better way to handle this case
    from netket.vqs import ExactState

    if isinstance(vstate, ExactState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    else:
        samples = vstate.samples
        pdf = None

    if mode is None:
        mode = choose_jacobian_mode(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            samples,
            mode=mode,
            holomorphic=holomorphic,
        )

    if chunk_size is None and hasattr(vstate, "chunk_size"):
        chunk_size = vstate.chunk_size

    shift, offset = to_shift_offset(diag_shift, diag_scale)

    jacobians = nkjax.jacobian(
        vstate._apply_fun,
        vstate.parameters,
        samples.reshape(-1, samples.shape[-1]),
        vstate.model_state,
        mode=mode,
        pdf=pdf,
        chunk_size=chunk_size,
        dense=True,
        center=True,
    )

    if offset is not None:
        ndims = 1 if mode != "complex" else 2
        jacobians, scale = rescale(jacobians, offset, ndims=ndims)
    else:
        scale = None

    pars_struct = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vstate.parameters
    )

    return QGTJacobianDenseT(
        O=jacobians,
        scale=scale,
        mode=mode,
        _params_structure=pars_struct,
        diag_shift=shift,
        **kwargs,
    )


@struct.dataclass
class QGTJacobianDenseT(LinearOperator):
    """
    Semi-lazy representation of an S Matrix behaving like a linear operator.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
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
        - "real": for real-valued R->R and C->R Ansätze, splits the complex inputs
                  into real and imaginary part.
        - "complex": for complex-valued R->C and C->C Ansätze, splits the complex
                  inputs and outputs into real and imaginary part
        - "holomorphic": for any Ansätze. Does not split complex values.
        - "auto": autoselect real or complex.
    """

    _in_solve: bool = struct.field(pytree_node=False, default=False)
    """Internal flag used to signal that we are inside the _solve method and matmul should
    not take apart into real and complex parts the other vector"""

    _params_structure: PyTree = struct.field(pytree_node=False, default=Uninitialized)

    def __matmul__(self, vec: Union[PyTree, jnp.ndarray]) -> Union[PyTree, jnp.ndarray]:
        return _matmul(self, vec)

    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None) -> PyTree:
        return _solve(self, solve_fun, y, x0=x0)

    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        return _to_dense(self)

    def __repr__(self):
        return (
            f"QGTJacobianDense(diag_shift={self.diag_shift}, "
            f"scale={self.scale}, mode={self.mode})"
        )


########################################################################################
#####                                  QGT Logic                                   #####
########################################################################################


@jax.jit
def _matmul(
    self: QGTJacobianDenseT, vec: Union[PyTree, jnp.ndarray]
) -> Union[PyTree, jnp.ndarray]:

    unravel = None
    if not hasattr(vec, "ndim") and not self._in_solve:
        check_valid_vector_type(self._params_structure, vec)
        vec, unravel = nkjax.tree_ravel(vec)

    # Real-imaginary split RHS in R→R and R→C modes
    reassemble = None
    if self.mode != "holomorphic" and not self._in_solve:
        vec, reassemble = vec_to_real(vec)

    if self.scale is not None:
        vec = vec * self.scale

    result = mat_vec(vec, self.O, self.diag_shift)

    if self.scale is not None:
        result = result * self.scale

    if reassemble is not None:
        result = reassemble(result)

    if unravel is not None:
        result = unravel(result)

    return result


@jax.jit
def _solve(
    self: QGTJacobianDenseT, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None
) -> PyTree:
    if not hasattr(y, "ndim"):
        check_valid_vector_type(self._params_structure, y)

    # Ravel input PyTrees, record unravelling function too
    y, unravel = nkjax.tree_ravel(y)

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

    return unravel(out), info


@jax.jit
def _to_dense(self: QGTJacobianDenseT) -> jnp.ndarray:
    if self.scale is None:
        O = self.O
        diag = jnp.eye(self.O.shape[-1])
    else:
        O = self.O * self.scale[jnp.newaxis, :]
        diag = jnp.diag(self.scale**2)

    # concatenate samples with real/Imaginary dimension
    O = O.reshape(-1, O.shape[-1])
    return mpi.mpi_sum_jax(O.conj().T @ O)[0] + self.diag_shift * diag
