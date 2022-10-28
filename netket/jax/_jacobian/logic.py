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
    model_state: Optional[PyTree],
    params: PyTree,
    samples: Array,
    *,
    mode: str,
    pdf=None,
    chunk_size: int = None,
    center: bool = False,
    dense: bool = False,
) -> PyTree:
    """
    compute ΔOⱼₖ = Oⱼₖ - ⟨Oₖ⟩ = ∂/∂pₖ ln Ψ(σⱼ) - ⟨∂/∂pₖ ln Ψ⟩
    divided by √n

    In a somewhat intransparent way this also internally splits all parameters to real
    in the 'real' and 'complex' modes (for C→R, R&C→R, R&C→C and general C→C) resulting in the respective ΔOⱼₖ
    which is only compatible with split-to-real pytree vectors

    Args:
        apply_fun: The forward pass of the Ansatz
        params : a pytree of parameters p
        samples : an array of (n in total) batched samples σ
        model_state: untrained state parameters of the model
        mode: differentiation mode, must be one of 'real', 'complex', 'holomorphic'
        pdf: |ψ(x)|^2 if exact optimization is being used else None
        chunk_size: an int specifying the size of the chunks the gradient should be computed in (default: None)

    Returns:
        if not rescale_shift:
            a pytree representing the centered jacobian of ln Ψ evaluated at the samples σ, divided by √n;
            None
        else:
            the same pytree, but the entries for each parameter normalised to unit norm;
            pytree containing the norms that were divided out (same shape as params)

    """
    # un-batch the samples
    samples = samples.reshape((-1, samples.shape[-1]))

    # pre-apply the model state
    def forward_fn(W, σ):
        return apply_fun({"params": W, **model_state}, σ)

    if dense:
        jac = jacobian_dense
    else:
        jac = jacobian_pytree

    if mode == "real":
        split_complex_params = True  # convert C→R and R&C→R to R→R
        jacobian_fun = jac.jacobian_real_holo_fun
    elif mode == "complex":
        split_complex_params = True  # convert C→C and R&C→C to R→C

        # avoid converting to complex and then back
        # by passing around the oks as a tuple of two pytrees representing the real and imag parts
        jacobian_fun = jac.jacobian_cplx_fun
    elif mode == "holomorphic":
        split_complex_params = False
        jacobian_fun = jac.jacobian_real_holo_fun
    else:
        raise NotImplementedError(
            'Differentiation mode should be one of "real", "complex", or "holomorphic", got {}'.format(
                mode
            )
        )

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
