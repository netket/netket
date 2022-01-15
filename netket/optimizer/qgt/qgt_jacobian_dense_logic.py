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
from functools import partial, wraps

import numpy as np
import jax
from jax import numpy as jnp

from netket.stats import subtract_mean
from netket.utils.types import PyTree, Array
from netket.utils import mpi
import netket.jax as nkjax

from .qgt_jacobian_pytree_logic import (
    single_sample,
)


# Utilities for splitting real and imaginary part
def _to_re(x):
    if jnp.iscomplexobj(x):
        return x.real
        # TODO find a way to make it a nop?
        # return jax.vmap(lambda y: jnp.array((y.real, y.imag)))(x)
    else:
        return x


def _to_im(x):
    if jnp.iscomplexobj(x):
        return x.imag
        # TODO find a way to make it a nop?
        # return jax.vmap(lambda y: jnp.array((y.real, y.imag)))(x)
    else:
        return None


def _tree_to_reim(x):
    return (jax.tree_map(_to_re, x), jax.tree_map(_to_im, x))


def _tree_reassemble_complex(x, target, fun=_tree_to_reim):
    (res,) = jax.linear_transpose(fun, target)(x)
    return nkjax.tree_conj(res)


# This function differs from tree_to_real because once ravelled
# the real parts are all contiguous and then stacked on top of
# the complex parts.
def tree_to_reim(pytree: PyTree) -> Tuple[PyTree, Callable]:
    """Replace the PyTree with a tuple of PyTrees with the same
    structure but containing only the real and imaginary part
    of the leaves. Real leaves are not duplicated.

    Args:
      pytree: a pytree to convert to real

    Returns:
      A pair where the first element is the tuple of pytrees,
      and the second element is a callable for converting back the tuple of
      pytrees to a complex pytree of the same structure as the input pytree.
    """
    return _tree_to_reim(pytree), partial(_tree_reassemble_complex, target=pytree)


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
    out, reassemble = nkjax.tree_to_real(vec)

    if nkjax.is_complex(vec):
        re, im = out

        out = jnp.concatenate([re, im], axis=0)

        def reassemble_concat(x):
            x = tuple(jnp.split(x, 2, axis=0))
            return reassemble(x)

    else:
        reassemble_concat = reassemble

    return out, reassemble_concat


#  TODO thos 3 functions are the same as those in qgt_jac_pytree_logic.py
# but without the vmap.
# we should cleanup and de-duplicate the code.


def jacobian_real_holo(forward_fn: Callable, params: PyTree, σ: Array) -> PyTree:
    """Calculates one Jacobian entry.
    Assumes the function is R→R or holomorphic C→C, so single vjp is enough.

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        σ : a single sample (vector)

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """
    y, vjp_fun = jax.vjp(single_sample(forward_fn), params, σ)
    res, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    return res


def _jacobian_cplx(
    forward_fn: Callable, params: PyTree, samples: Array, _build_fn: Callable
) -> PyTree:
    """Calculates one Jacobian entry.
    Assumes the function is R→C, backpropagates 1 and -1j

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        σ : a single sample (vector)

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """
    y, vjp_fun = jax.vjp(single_sample(forward_fn), params, samples)
    gr, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    gi, _ = vjp_fun(np.array(-1.0j, dtype=jnp.result_type(y)))
    return _build_fn(gr, gi)


@partial(wraps(_jacobian_cplx))
def jacobian_cplx(
    forward_fn, params, samples, _build_fn=partial(jax.tree_multimap, jax.lax.complex)
):
    return _jacobian_cplx(forward_fn, params, samples, _build_fn)


def stack_jacobian_tuple(ok_re_im):
    """
    stack the real and imaginary parts of ΔOⱼₖ along a new axis.
    First all the real part then the imaginary part.

    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]

    Args:
        ok_re_im : a tuple (ΔOᵣ, ΔOᵢ) of two PyTrees representing the real and imag part of ΔOⱼₖ
    """
    re, im = ok_re_im

    re_dense = ravel(re)
    im_dense = ravel(im)
    res = jnp.stack([re_dense, im_dense], axis=0)

    return res


def ravel(x: PyTree) -> Array:
    """
    shorthand for tree_ravel
    """
    dense, _ = nkjax.tree_ravel(x)
    return dense


dense_jacobian_real_holo = nkjax.compose(ravel, jacobian_real_holo)
dense_jacobian_cplx = nkjax.compose(
    stack_jacobian_tuple, partial(jacobian_cplx, _build_fn=lambda *x: x)
)


def _rescale(centered_oks):
    """
    compute ΔOₖ/√Sₖₖ and √Sₖₖ
    to do scale-invariant regularization (Becca & Sorella 2017, pp. 143)
    Sₖₗ/(√Sₖₖ√Sₗₗ) = ΔOₖᴴΔOₗ/(√Sₖₖ√Sₗₗ) = (ΔOₖ/√Sₖₖ)ᴴ(ΔOₗ/√Sₗₗ)
    """
    scale = (
        mpi.mpi_sum_jax(
            jnp.sum((centered_oks * centered_oks.conj()).real, axis=0, keepdims=True)
        )[0]
        ** 0.5
    )
    centered_oks = jnp.divide(centered_oks, scale)
    scale = jnp.squeeze(scale, axis=0)
    return centered_oks, scale


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
        chunk_size: an int specfying the size of the chunks degradient should be computed in (default: None)

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
        params, reassemble = tree_to_reim(params)

        def f(W, σ):
            return forward_fn(reassemble(W), σ)

    else:
        f = forward_fn

    def gradf_fun(params, σ):
        gradf_dense = jacobian_fun(f, params, σ)
        return gradf_dense

    jacobians = nkjax.vmap_chunked(gradf_fun, in_axes=(None, 0), chunk_size=chunk_size)(
        params, samples
    )

    n_samp = samples.shape[0] * mpi.n_nodes
    centered_oks = subtract_mean(jacobians, axis=0) / np.sqrt(n_samp)

    centered_oks = centered_oks.reshape(-1, centered_oks.shape[-1])

    if rescale_shift:
        return _rescale(centered_oks)
    else:
        return centered_oks, None
