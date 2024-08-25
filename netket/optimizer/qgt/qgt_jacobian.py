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


import jax

from netket.utils import timing
from netket.utils.api_utils import partial_from_kwargs
from netket import jax as nkjax
from netket.nn import split_array_mpi

from .qgt_jacobian_dense import QGTJacobianDenseT
from .qgt_jacobian_pytree import QGTJacobianPyTreeT
from .qgt_jacobian_common import (
    to_shift_offset,
    rescale,
)


@timing.timed
def QGTJacobian_DefaultConstructor(
    apply_fun,
    parameters,
    model_state,
    samples,
    pdf=None,
    *,
    dense: bool,
    mode: str | None = None,
    holomorphic: bool | None = None,
    diag_shift: float | None = 0.0,
    diag_scale: float | None = None,
    chunk_size: int | None = None,
    **kwargs,
) -> QGTJacobianDenseT | QGTJacobianPyTreeT:
    """
    Construct a :class:`QGTJacobianDenseT` or :class:`QGTJacobianPyTreeT`
    starting from the definition of a variational state.

    This type of the output is determined by the `dense` boolean keyword argument,
    which is mandatory.

    The `pdf` argument can be used to:
    - perform a full-summation calculation, where pdf is a probability density
      that sums to 1.
    - Similarly, it can be `jnp.ones_like(samples.shape[:-1])/Ns` to return
      the same result as `pdf = None`.
    - reweight individual samples a 'la importance sampling, where the pdf
      should now be the importance reweight factor times `1/Ns`.

    Other arguments are as other jacobian QGT constructors, documented below.

    This constructor is separated as it can be used from packages extending
    NetKet.
    """
    if mode is not None and holomorphic is not None:
        raise ValueError("Cannot specify both `mode` and `holomorphic`.")

    if mode is None:
        mode = nkjax.jacobian_default_mode(
            apply_fun,
            parameters,
            model_state,
            samples,
            holomorphic=holomorphic,
        )

    if pdf is not None:
        if not pdf.shape == samples.shape[:-1]:
            raise ValueError(
                "The shape of pdf must match the shape of the samples, "
                f"instead you provided (pdf.shape={pdf.shape}) != "
                f"(samples.shape={samples.shape[:-1]})"
            )
        if pdf.ndim >= 2:
            pdf = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(pdf, 0, 2)

    if samples.ndim >= 3:
        # use jit so that we can do it on global shared array
        samples = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(samples, 0, 2)

    jac_mode = mode
    if mode == "imag":
        # Imaginary mode is a specificity of the QGT, but it requires the standard complex-mode
        # jacobian to be computed.
        jac_mode = "complex"

    jacobians = nkjax.jacobian(
        apply_fun,
        parameters,
        samples,
        model_state,
        mode=jac_mode,
        pdf=pdf,
        chunk_size=chunk_size,
        dense=dense,
        center=True,
        _sqrt_rescale=True,
    )
    shift, offset = to_shift_offset(diag_shift, diag_scale)

    if offset is not None:
        ndims = 1 if (mode != "complex" and mode != "imag") else 2
        jacobians, scale = rescale(jacobians, offset, ndims=ndims)
    else:
        scale = None

    pars_struct = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )

    QGT_T = QGTJacobianDenseT if dense else QGTJacobianPyTreeT
    return QGT_T(
        O=jacobians,
        scale=scale,
        mode=mode,
        _params_structure=pars_struct,
        diag_shift=shift,
        **kwargs,
    )


@partial_from_kwargs(exclusive_arg_names=(("mode", "holomorphic")))
def QGTJacobianDense(
    vstate,
    *,
    mode: str | None = None,
    holomorphic: bool | None = None,
    diag_shift: float | None = 0.0,
    diag_scale: float | None = None,
    chunk_size: int | None = None,
    **kwargs,
) -> QGTJacobianDenseT:
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
    # TODO: Find a better way to handle this case
    from netket.vqs import FullSumState

    if isinstance(vstate, FullSumState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    else:
        samples = vstate.samples
        pdf = None

    if chunk_size is None:
        chunk_size = getattr(vstate, "chunk_size", None)

    return QGTJacobian_DefaultConstructor(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples,
        pdf=pdf,
        dense=True,
        mode=mode,
        holomorphic=holomorphic,
        diag_shift=diag_shift,
        diag_scale=diag_scale,
        chunk_size=chunk_size,
        **kwargs,
    )


@partial_from_kwargs(exclusive_arg_names=(("mode", "holomorphic")))
def QGTJacobianPyTree(
    vstate,
    *,
    mode: str | None = None,
    holomorphic: bool | None = None,
    diag_shift: float | None = 0.0,
    diag_scale: float | None = None,
    chunk_size: int | None = None,
    **kwargs,
) -> QGTJacobianPyTreeT:
    """
    Semi-lazy representation of an S Matrix where the Jacobian O_k is precomputed
    and stored as a PyTree.

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
    # TODO: Find a better way to handle this case
    from netket.vqs import FullSumState

    if isinstance(vstate, FullSumState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    else:
        samples = vstate.samples
        pdf = None

    if chunk_size is None:
        chunk_size = getattr(vstate, "chunk_size", None)

    return QGTJacobian_DefaultConstructor(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples,
        dense=False,
        pdf=pdf,
        mode=mode,
        holomorphic=holomorphic,
        diag_shift=diag_shift,
        diag_scale=diag_scale,
        chunk_size=chunk_size,
        **kwargs,
    )
