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

from netket.stats import subtract_mean, sum as sum_mpi
from netket.utils import mpi
from netket.utils.types import Array, Callable, PyTree
from netket.jax import (
    tree_to_real,
    vmap_chunked,
)

from . import jacobian_dense
from . import jacobian_pytree


@partial(
    jax.jit, static_argnames=("apply_fun", "mode", "chunk_size", "center", "dense")
)
def jacobian(
    apply_fun: Callable,
    params: PyTree,
    samples: Array,
    model_state: Optional[PyTree] = None,
    *,
    mode: str,
    pdf: Array = None,
    chunk_size: int = None,
    center: bool = False,
    dense: bool = False,
) -> PyTree:
    r"""
    Computes the jacobian of a NN model with respect to its parameters. This function differs from
    :ref:`jax.jac_bwd` because it supports models with both real and complex parameters, as well as
    non-holomorphic models.

    In the context of NQS, if you pass the log-wavefunction to to this function, it will compute the
    log-derivative of the wavefunction with respect to the parameters, i.e. the matrix commonly known
    as:

    .. math::

        O_k(\sigma) = \frac{\partial \ln \Psi(\sigma)}{\partial \theta_k}


    This function has three modes of operation that must be specified through the ``mode`` keyword-argument:
        - ``mode="real"``: The jacobian that is returned is real. The Imaginary part of
            :math:`\ln\Psi(\sigma)` is discarded if present. This mode is useful for models describing
            real-valued states with a sign. This coincides with the :math:`O_k(\sigma)` matrix for real-valued,
            real-output models.
        - ``mode="complex"``: The jacobian that is returned is complex. This mode  returns the standard
            :math:`O_k(\sigma)` matrix for real-parameters, complex-output models. If your model has complex
            parameters and it is not holomorphic, you should use this mode as well. In that case, it will
            split the jacobian and conjugate-jacobian into two different objects by splitting the real
            and imaginary part of the parameters.
        - ``mode="holomorphic"``: returns correct results only if your model is holomorphic. Works like
            ``mode="real"``, but returns a complex valued jacobian.

    The returned jacobian has the same PyTree structure as the parameters, with an additional leading
    dimension equal to the number of samples if ``mode=real/holomorphic`` or if you have real-valued parameters
    and use ``mode=complex``. If you have complex-valued parameters and use ``mode=complex``, the returned
    pytree will have two leading dimensions, the first iterating along the samples, and the second
    with size 2, iterating along the real and imaginary part of the parameters (essentially giving the
    jacobian and conjugate-jacobian).

    If dense is True, the returned jacobian is a dense matrix, that is somewhat similar to what would be
    obtained by calling ``jax.vmap(jax.grad(apply_fun))(parameters)``.

    In a somewhat intransparent way this also internally splits all parameters to real
    in the 'real' and 'complex' modes (for C→R, R&C→R, R&C→C and general C→C) resulting in the respective ΔOⱼₖ
    which is only compatible with split-to-real pytree vectors

    Args:
        apply_fun: The forward pass of the Ansatz
        model_state: untrained state parameters of the model
        params : a pytree of parameters p
        samples : an array of (n in total) batched samples σ
        mode: differentiation mode, must be one of 'real', 'complex', 'holomorphic', `real` as described above.
        pdf: |ψ(x)|^2 if exact optimization is being used else None
        chunk_size: an int specifying the size of the chunks the gradient should be computed in (default: None)
        center: a boolean specifying if the jacobian should be centered.

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
            'Differentiation mode should be one of "real", "complex", or "holomorphic", got {}'.format(
                mode
            )
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
    jacobians = vmap_chunked(
        jacobian_fun, in_axes=(None, None, 0), chunk_size=chunk_size
    )(f, params, samples)

    if pdf is None:
        sqrt_n_samp = math.sqrt(samples.shape[0] * mpi.n_nodes)  # maintain weak type
        if center:
            jacobians = jax.tree_map(
                lambda x: subtract_mean(x, axis=0) / sqrt_n_samp, jacobians
            )

    else:
        if center:
            jacobians_avg = jax.tree_map(
                partial(sum_mpi, axis=0), _multiply_by_pdf(jacobians, pdf)
            )
            jacobians = jax.tree_map(lambda x, y: x - y, jacobians, jacobians_avg)

        jacobians = _multiply_by_pdf(jacobians, jnp.sqrt(pdf))

    return jacobians


def _multiply_by_pdf(oks, pdf):
    """
    Computes  O'ⱼ̨ₖ = Oⱼₖ pⱼ .
    Used to multiply the log-derivatives by the probability density.
    """

    return jax.tree_map(
        lambda x: jax.lax.broadcast_in_dim(pdf, x.shape, (0,)) * x,
        oks,
    )
