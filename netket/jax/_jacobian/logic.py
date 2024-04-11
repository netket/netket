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

from typing import Optional
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from netket.stats import subtract_mean, sum as sum_mpi
from netket.utils import mpi, timing
from netket.utils.types import Array, Callable, PyTree
from netket.jax import (
    tree_to_real,
    vmap_chunked,
)

from . import jacobian_dense
from . import jacobian_pytree


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "apply_fun",
        "mode",
        "chunk_size",
        "center",
        "dense",
        "_sqrt_rescale",
    ),
)
def jacobian(
    apply_fun: Callable,
    params: PyTree,
    samples: Array,
    model_state: Optional[PyTree] = None,
    *,
    mode: str,
    pdf: Array = None,
    chunk_size: Optional[int] = None,
    center: bool = False,
    dense: bool = False,
    _sqrt_rescale: bool = False,
) -> PyTree:
    r"""
    Computes the jacobian of a NN model with respect to its parameters. This function
    differs from :func:`jax.jacrev` because it supports models with both real and
    complex parameters, as well as non-holomorphic models.

    In the context of NQS, if you pass the log-wavefunction to to this function, it
    will compute the log-derivative of the wavefunction with respect to the
    parameters, i.e. the matrix commonly known as:

    .. math::

        O_k(\sigma) = \frac{\partial \ln \Psi(\sigma)}{\partial \theta_k}


    This function has three modes of operation that must be specified through
    the :code:`mode` keyword-argument:

    - :code:`mode="real"`: Which works for real-valued functions with real-
      valued parameters, or truncating the imaginary part of the function.
    - :code:`mode="complex"` which always returns the correct result, but
      results in redundant computations if the function is holomorphic.
      For functions of real-parameters but complex output returns the
      derivatives of the real and imaginary part concatenated. If the
      parameters are complex the derivatives w.r.t. the real and imaginary
      part of the parameters are split into two different jacobians.
    - :code:`mode="holomorphic"` for complex-valued, complex parametrs
      holomorphic functions.

    Args:
        apply_fun: The function for which the jacobian should be computed. It
            must have the signature :code:`f: PyTree, Array -> Array` where the
            first :class:`PyTree` are the parameters with respect to which the
            jacobian will be computed, while the second argument is not
            differentiated. The second argument (samples) should be a 2D batch
            of inputs.
        params : The PyTree of parameters (:math:`\theta` in the equations),
            with repsect to which the jacobian will computed.
        samples : A batch of samples (:math:`\sigma` in the equations), encoded
            in a 2D matrix where the first dimension is the batch dimension and
            the latter dimension encodes the different degrees of freedom. The
            gradient is not computed with respect to this argument.
        model_state: Optional model variables that are not trained/differentiated.
            See the jax documentation to understand how those are used.
        mode: differentiation mode, must be one of `real`, `complex` or
            `holomorphic` as quickly described above. For a detailed explanation,
            read the detailed discussion below.
        pdf: Optional coefficient that is used to multiply every row of the Jacobian.
            When performing calculations in full-summation, this can be used to
            multiply every row by :math:`|\psi(\sigma)|^2`, which is needed to
            compute the correct average.
        chunk_size: Optional integer specifying the maximum number of samples for
            which the gradient is simulataneously computed. Low-values will
            require lower amounts of memory, but might increase computational cost
            (chunking is disabled by default).
        center: a boolean specifying if the jacobian should be centered (disabled
            by default).
        dense: a boolean flag (disabled by default) to specify if the jacobian
            should be raveled to a contiguous dense array. For *real* and
            *holomorphic* mode this will return a 2D matrix where the first
            dimension matches the number of samples (the first axis of **samples**),
            while the second dimension will match the total number of parameters.
            This raveling is equivalent to :func:`jax.vmap` of
            :func:`netket.jax.tree_ravel`,
            :code:`jax.vmap(nk.jax.tree_ravel, nk.jax.jacobian(...))`.
            If using **complex** mode with real parameters the returned tensor
            has 3 dimensions, where the first and last match the other modes while
            the middle one has size 2, and encodes the gradient of the real and
            imaginary part of **apply_fun**.
            If using **complex** mode with complex parameters, the returned
            tensor has 3 dimensions, where the first has the number of samples,
            the second has size 2 as described above, and the last has twice the
            number of parameters, where the first :math:`N_\text{pars}` elements
            are the derivatives wrt the real part of the parameters, while the
            second :math:`N_\text{pars}` elements are the derivatives wrt the
            imaginary part of the paramters.
        _sqrt_rescale: **internal flag** (do not rely on it) a boolean flag
            (disabled by default). If enabled, the jacobian is rescaled by
            :math:`1/\sqrt{N_\text{samples}}` to match the scaling emerging in
            some use-cases such when building the Quantum Geometric Tensor.
            If a pdf is specified, the scaling will instead be
            :math:`\sqrt{pdf_i}`. This flag is temporary and internal and might
            be discontinued at any point in the future. Do not use it.


    Extra details of the different modes are given below:

    Real-valued mode (:code:`mode='real'`)
    --------------------------------------

    This mode should be used for functions with real
    output or if you wish to truncate the imaginary part of the jacobian.
    Practically, it computes the Jacobian defined as

    .. math::

       O_k(\sigma) = \frac{\partial \ln\Re[\Psi(\sigma)]}{\partial \Re[\theta_k]}

    and it should return a result roughly equivalent to the following listing:

    .. code:: python

      samples = samples.reshape(-1, samples.shape[-1])
      parameters = jax.tree_util.tree_map(lambda x: x.real, parameters)
      O_k = jax.jacrev(lambda pars: logpsi(pars, samples).real, parameters)

    The jacobian that is returned is a PyTree with the same shape
    as :code:`parameters`, with real data type.
    The Imaginary part of :math:`\ln\Psi(\sigma)` is discarded if present.
    This mode is useful for models describing real-valued states with a sign.
    This coincides with the :math:`O_k(\sigma)` matrix for real-valued,
    real-output models.

    Complex-valued, non-holomorphic mode (:code:`mode='complex'`)
    -------------------------------------------------------------

    This function computes all the information necessary
    to reconstruct the Jacobian and potentially the conjugate-Jacobian that is
    non-zero for non-holomorphic functions. It should be used for:

    - complex-valued functions with real parameters, of which we do not want
      to truncate the imaginary part;
    - complex-valued functions with mixed real and complex parameters, which
      are therefore not-holomorphic;
    - complex-valued functions with complex parameters which are not
      holomorphic (if the function is holomoprhic, the results will be correct
      but the returned data will be redundant);

    **If all parameters** :math:`\theta_k` **are real**, this mode returns the
    derivatives of the real and imaginary part of the function,

    .. math::

       O^{r}_k(\sigma) = \frac{\partial \ln\Re[\Psi(\sigma)]}{\partial \theta_k}
       \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,
       O^{i}_k(\sigma) = \frac{\partial \ln\Im[\Psi(\sigma)]}{\partial \theta_k}

    where :math:`O^{r}_k(\sigma)` and :math:`O^{i}_k(\sigma)` are real-valued pytrees
    with the same shape as the original parameters. In practice,
    it should return a result roughly equivalent to the following listing:

    .. code:: python

      samples = samples.reshape(-1, samples.shape[-1])
      Or_k = jax.jacrev(lambda pars: logpsi(pars, samples).real, parameters)
      Oi_k = jax.jacrev(lambda pars: logpsi(pars, samples).imag, parameters)
      O_k = jax.tree_util.tree_map(lambda jr, ji: jnp.concatenate([jr, ji]], axis=1),
                                                        Or_k, Oi_k)

    As both :code:`Or_k` and :code:`Oi_k` are real, instead of concatenating we
    could also construct the full complex Jacobian. However, we chose not to
    do this for performance reason, but the downstream user is free to do it if
    he wishes.

    If you wish to get the complex jacobian in the case of real parameters, it is
    possible to define

    .. math::

         O_k(\sigma) =  O^{r}_k(\sigma) + i O^{i}_k(\sigma)

    which is now complex-valued. In code, this is equivalent to

    .. code:: python

      O_k_cmplx = jax.tree_util.tree_map(lambda jri: jri[:, 0, :] + 1j* jri[:, 1, :], O_k)


    **If some parameters** :math:`\theta_k` **are complex**, this mode splits the
    :math:`N` complex parameters into :math:`2N` real parameters, where the first
    block of :math:`N` parameters correspond to the real parts and the latter block
    to the imaginary part, and then follows the logic discussed above.

    In formulas, this can be seen as defining the vector of :math:`2N` real parameters

    .. math::

        \tilde\theta = (\Re[\theta], \Im[\theta])

    and then computing the same quantities as above

    .. math::

       O^{r}_k(\sigma) = \frac{\partial \ln\Re[\Psi(\sigma)]}{\partial \tilde\theta_k]}
       \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,
       O^{i}_k(\sigma) = \frac{\partial \ln\Im[\Psi(\sigma)]}{\partial \tilde\theta_k]}

    where now those objects have twice the number of elements as the parameters.
    In practice, it should return a result roughly equivalent to the following listing:

    .. code:: python

      samples = samples.reshape(-1, samples.shape[-1])
      # tree_to_real splits the parameters in a tuple like
      # {'real': jax.tree.map(jnp.real, pars), 'imag': jax.tree.map(jnp.imag, pars)}
      pars_real, reconstruct = nk.jax.tree_to_real(parameters)
      Or_k = jax.jacrev(lambda pars_re: logpsi(reconstruct(pars_re), samples).real,
                        pars_real)
      Oi_k = jax.jacrev(lambda pars_re: logpsi(reconstruct(pars_re), samples).imag,
                        pars_real)
      O_k = jax.tree_util.tree_map(lambda jr, ji: jnp.concatenate([jr, ji]], axis=1),
                                                        Or_k, Oi_k)

    This code is also valid if all parameters are real, in which case :code:`O_k.real`
    is identical to what was described above. Otherwise, :code:`O_k.imag` contains
    the derivative w.r.t. the imaginary part of the parameters which are complex.
    Every element in :code:`O_k` has the shape :code:`(N_s, 2, ...)` where
    :math:`N_{s}` is the number of samples and 2 arises from the derivatives wrt the
    real and imaginary parts.

    Holomorphic mode (:code:`mode='holomorphic'`)
    ---------------------------------------------

    This function computes the gradient with respect
    to the complex parameters :math:`\theta`. It can only be applied to functions
    whose parameters are all complex-valued, and which are holomorphic (they satisfy
    `Cauchy-Riemann equations <https://en.wikipedia.org/wiki/Cauchy–Riemann_equations>`_,
    which can be numerically checked with :func:`~netket.utils.is_probably_holomorphic`).
    This function is roughly equivalent to

    .. code:: python

      samples = samples.reshape(-1, samples.shape[-1])
      O_k = jax.jacrev(lambda pars: logpsi(pars, samples), parameters, holomorphic=True)

    If the function is not holomorphic the result will be numerically wrong.


    The returned jacobian has the same PyTree structure as the parameters, with an
    additional leading dimension equal to the number of samples if
    :code:`mode=real/holomorphic` or if you have real-valued parameters
    and use :code:`mode=complex`. If you have complex-valued parameters
    and use :Code:`mode=complex`, the returned pytree will have two leading
    dimensions, the first iterating along the samples, and the second with size 2,
    iterating along the real and imaginary part of the parameters (essentially
    giving the jacobian and conjugate-jacobian).

    If dense is True, the returned jacobian is a dense matrix, that is somewhat
    similar to what would be obtained by calling
    :code:`jax.vmap(jax.grad(apply_fun))(parameters)`.

    In a somewhat intransparent way this also internally splits all parameters
    to real in the 'real' and 'complex' modes (for C→R, R&C→R, R&C→C and
    general C→C) resulting in the respective ΔOⱼₖ which is only compatible with
    split-to-real pytree vectors

    """
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array")

    if model_state is None:
        model_state = {}

    if dense:
        jac_type = jacobian_dense
    else:
        jac_type = jacobian_pytree

    if mode == "real":
        split_complex_params = True  # convert C→R and R&C→R to R→R
        jacobian_fun = jac_type.jacobian_real_holo_fun
    elif mode == "complex":
        split_complex_params = True  # convert C→C and R&C→C to R→C

        # avoid converting to complex and then back
        # by passing around the oks as a tuple of two pytrees representing the real and imag parts
        jacobian_fun = jac_type.jacobian_cplx_fun
    elif mode == "holomorphic":
        split_complex_params = False
        jacobian_fun = jac_type.jacobian_real_holo_fun
    else:
        raise NotImplementedError(
            "Differentiation mode should be one of 'real', "
            f"'complex', or 'holomorphic', got {mode}"
        )

    # pre-apply the model state
    forward_fn = lambda W, σ: apply_fun({"params": W, **model_state}, σ)

    if split_complex_params:
        # doesn't do anything if the params are already real
        params, reassemble = tree_to_real(params)
        f = lambda W, σ: forward_fn(reassemble(W), σ)
    else:
        f = forward_fn

    # jacobians is a tree with leaf shapes:
    # - (n_samples, 2, ...) if mode complex, holding the real and imaginary jacobian
    # - (n_samples, ...) if mode real/holomorphic
    # here we wrap f with a Partial since the shard_map inside vmap_chunked
    # does not support non-array arguments
    jacobians = vmap_chunked(
        jacobian_fun, in_axes=(None, None, 0), chunk_size=chunk_size
    )(Partial(f), params, samples)

    if pdf is None:
        if center:
            jacobians = jax.tree_util.tree_map(
                lambda x: subtract_mean(x, axis=0), jacobians
            )

        if _sqrt_rescale:
            sqrt_n_samp = math.sqrt(
                samples.shape[0] * mpi.n_nodes
            )  # maintain weak type
            jacobians = jax.tree_util.tree_map(lambda x: x / sqrt_n_samp, jacobians)

    else:
        if center:
            jacobians_avg = jax.tree_util.tree_map(
                partial(sum_mpi, axis=0), _multiply_by_pdf(jacobians, pdf)
            )
            jacobians = jax.tree_util.tree_map(
                lambda x, y: x - y, jacobians, jacobians_avg
            )

        if _sqrt_rescale:
            jacobians = _multiply_by_pdf(jacobians, jnp.sqrt(pdf))

    return jacobians


def _multiply_by_pdf(oks, pdf):
    """
    Computes  O'ⱼ̨ₖ = Oⱼₖ pⱼ .
    Used to multiply the log-derivatives by the probability density.
    """

    return jax.tree_util.tree_map(
        lambda x: jax.lax.broadcast_in_dim(pdf, x.shape, (0,)) * x,
        oks,
    )
