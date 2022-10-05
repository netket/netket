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

from typing import Callable, Optional, Tuple
from functools import partial
import math

import jax
from jax import numpy as jnp

from netket.stats import subtract_mean
from netket.utils.types import PyTree, Array
from netket.utils import mpi
import netket.jax as nkjax

from .qgt_jacobian_pytree_logic import (
    jacobian_real_holo,
    jacobian_cplx,
    _rescale_leaf as _rescale,
)

from netket.jax.utils import RealImagTuple


def vec_to_real(vec: Array) -> Tuple[Array, Callable]:
    """
    If the input vector is real, splits the vector into real
    and imaginary parts and concatenates them along the 0-th
    axis.

    It is equivalent to changing the complex storage from AOS
    to SOA.

    Args:
        vec: a dense vector
    """
    if jnp.iscomplexobj(vec):
        out, reassemble = nkjax.tree_to_real(vec)
        out = jnp.concatenate([out.real, out.imag], axis=0)

        def reassemble_concat(x):
            x = RealImagTuple(jnp.split(x, 2, axis=0))
            return reassemble(x)

        return out, reassemble_concat

    else:
        return vec, lambda x: x


def ravel(x: PyTree) -> Array:
    """
    shorthand for tree_ravel
    """
    dense, _ = nkjax.tree_ravel(x)
    return dense


def stack_jacobian_tuple(ok_re_im):
    """
    stack the real and imaginary parts of ΔOⱼₖ along a new axis.
    First all the real part then the imaginary part.

    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]

    Args:
        ok_re_im : a tuple (ΔOᵣ, ΔOᵢ) of two PyTrees representing the real and imag part of ΔOⱼₖ
    """
    re, im = ok_re_im
    return jnp.stack([ravel(re), ravel(im)], axis=0)


dense_jacobian_real_holo = nkjax.compose(ravel, jacobian_real_holo)
dense_jacobian_cplx = nkjax.compose(stack_jacobian_tuple, jacobian_cplx)


@partial(jax.jit, static_argnames=("apply_fun", "mode", "rescale_shift", "chunk_size"))
def prepare_centered_oks(
    apply_fun: Callable,
    params: PyTree,
    samples: Array,
    model_state: Optional[PyTree],
    mode: str,
    rescale_shift: bool,
    chunk_size: int = None,
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
        rescale_shift: whether scale-invariant regularisation should be used (default: True)
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

    if mode == "real":
        split_complex_params = True  # convert C→R and R&C→R to R→R
        jacobian_fun = dense_jacobian_real_holo
    elif mode == "complex":
        split_complex_params = True  # convert C→C and R&C→C to R→C
        # centered_jacobian_fun = compose(stack_jacobian, centered_jacobian_cplx)

        # avoid converting to complex and then back
        # by passing around the oks as a tuple of two pytrees representing the real and imag parts
        jacobian_fun = dense_jacobian_cplx
    elif mode == "holomorphic":
        split_complex_params = False
        jacobian_fun = dense_jacobian_real_holo
    else:
        raise NotImplementedError(
            'Differentiation mode should be one of "real", "complex", or "holomorphic", got {}'.format(
                mode
            )
        )

    # Stored as contiguous real stacked on top of contiguous imaginary (SOA)
    if split_complex_params:
        # doesn't do anything if the params are already real
        params, reassemble = nkjax.tree_to_real(params)

        def f(W, σ):
            return forward_fn(reassemble(W), σ)

    else:
        f = forward_fn

    # jacobians has shape:
    # - (n_samples, 2, n_pars) if mode complex, holding the real and imaginary jacobian
    # - (n_samples, n_pars) if mode real/holomorphic
    jacobians = nkjax.vmap_chunked(jacobian_fun, in_axes=(None, None, 0), chunk_size=chunk_size)(
        f, params, samples
    )

    n_samp = samples.shape[0] * mpi.n_nodes
    centered_jacs = subtract_mean(jacobians, axis=0) / math.sqrt(
        n_samp
    )  # maintain weak type!

    # here the jacobian is reshaped and the real/complex part are concatenated.
    centered_jacs = centered_jacs.reshape(-1, centered_jacs.shape[-1])

    if rescale_shift:
        return _rescale(centered_jacs)
    else:
        return centered_jacs, None
