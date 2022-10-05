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
import jax.flatten_util
import jax.numpy as jnp

import numpy as np

from netket.stats import subtract_mean, sum as sum_mpi
from netket.utils import mpi
from netket.utils.types import Array, Callable, PyTree, Scalar
from netket.jax import (
    tree_cast,
    tree_conj,
    tree_axpy,
    tree_to_real,
    compose,
    vmap_chunked,
)


# TODO better name and move it somewhere sensible
def single_sample(forward_fn):
    """
    A decorator to make the forward_fn accept a single sample
    """
    return lambda W, σ: forward_fn(W, σ[jnp.newaxis, :])[0]


def jacobian_real_holo(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    """Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→R or holomorphic C→C, so single grad is enough

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """

    y, vjp_fun = jax.vjp(lambda pars: single_sample(forward_fn)(pars, samples), params)
    (res,) = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    return res


def _jacobian_cplx(
    forward_fn: Callable,
    params: PyTree,
    samples: Array,
    *,
    _build_fn: Callable = partial(jax.tree_map, jax.lax.complex),
) -> PyTree:
    """Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→C, backpropagates 1 and -1j

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """

    y, vjp_fun = jax.vjp(lambda pars: single_sample(forward_fn)(pars, samples), params)
    (gr,) = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    (gi,) = vjp_fun(np.array(-1.0j, dtype=jnp.result_type(y)))
    return _build_fn(gr, gi)


jacobian_cplx = partial(_jacobian_cplx, _build_fn=lambda *x: x)


def _multiply_by_pdf(oks, pdf):
    """
    Computes  O'ⱼ̨ₖ = Oⱼₖ pⱼ .
    Used to multiply the log-derivatives by the probability density.
    """

    return jax.tree_map(
        lambda x: jax.lax.broadcast_in_dim(pdf, x.shape, (0,)) * x,
        oks,
    )


def stack_jacobian_tuple(centered_oks_re_im):
    """
    stack the real and imaginary parts of ΔOⱼₖ along the sample axis

    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]

    Args:
        centered_oks_re_im : a tuple (ΔOᵣ, ΔOᵢ) of two PyTrees representing the real and imag part of ΔOⱼₖ
    """
    return jax.tree_map(lambda re, im: jnp.stack([re, im], axis=0), *centered_oks_re_im)


def _rescale_leaf(centered_oks):
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


_rescale = partial(jax.tree_map, _rescale_leaf)


def _jvp(oks: PyTree, v: PyTree) -> Array:
    """
    Compute the matrix-vector product between the pytree jacobian oks and the pytree vector v
    """
    td = lambda x, y: jnp.tensordot(x, y, axes=y.ndim)
    t = jax.tree_map(td, oks, v)
    return jax.tree_util.tree_reduce(jnp.add, t)


def _vjp(oks: PyTree, w: Array) -> PyTree:
    """
    Compute the vector-matrix product between the vector w and the pytree jacobian oks
    """
    res = jax.tree_map(partial(jnp.tensordot, w, axes=w.ndim), oks)
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # MPI


def _mat_vec(v: PyTree, oks: PyTree) -> PyTree:
    """
    Compute ⟨O† O⟩v = ∑ₗ ⟨Oₖᴴ Oₗ⟩ vₗ
    """
    res = tree_conj(_vjp(oks, _jvp(oks, v).conjugate()))
    return tree_cast(res, v)


# ==============================================================================
# the logic above only works for R→R, R→C and holomorphic C→C
# here the other modes are converted


@partial(jax.jit, static_argnames=("apply_fun", "mode", "rescale_shift", "chunk_size"))
def prepare_centered_oks(
    apply_fun: Callable,
    params: PyTree,
    samples: Array,
    model_state: Optional[PyTree],
    mode: str,
    rescale_shift: bool,
    pdf=None,
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

    if mode == "real":
        split_complex_params = True  # convert C→R and R&C→R to R→R
        jacobian_fun = jacobian_real_holo
    elif mode == "complex":
        split_complex_params = True  # convert C→C and R&C→C to R→C

        # avoid converting to complex and then back
        # by passing around the oks as a tuple of two pytrees representing the real and imag parts
        jacobian_fun = compose(stack_jacobian_tuple, jacobian_cplx)
    elif mode == "holomorphic":
        split_complex_params = False
        jacobian_fun = jacobian_real_holo
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
        centered_jacs = jax.tree_map(
            lambda x: subtract_mean(x, axis=0) / sqrt_n_samp, jacobians
        )

    else:
        jacobians_avg = jax.tree_map(
            partial(sum_mpi, axis=0), _multiply_by_pdf(jacobians, pdf)
        )
        centered_jacs = jax.tree_map(lambda x, y: x - y, jacobians, jacobians_avg)

        centered_jacs = _multiply_by_pdf(centered_jacs, jnp.sqrt(pdf))
    if rescale_shift:
        return _rescale(centered_jacs)
    else:
        return centered_jacs, None


def mat_vec(v: PyTree, centered_oks: PyTree, diag_shift: Scalar) -> PyTree:
    """
    Compute (S + δ) v = 1/n ⟨ΔO† ΔO⟩v + δ v = ∑ₗ 1/n ⟨ΔOₖᴴΔOₗ⟩ vₗ + δ vₗ

    Only compatible with R→R, R→C, and holomorphic C→C
    for C→R, R&C→R, R&C→C and general C→C the parameters for generating ΔOⱼₖ should be converted to R,
    and thus also the v passed to this function as well as the output are expected to be of this form

    Args:
        v: pytree representing the vector v compatible with centered_oks
        centered_oks: pytree of gradients 1/√n ΔOⱼₖ
        diag_shift: a scalar diagonal shift δ
    Returns:
        a pytree corresponding to the sr matrix-vector product (S + δ) v
    """
    return tree_axpy(diag_shift, v, _mat_vec(v, centered_oks))
