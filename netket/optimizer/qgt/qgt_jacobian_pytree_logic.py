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
from netket.jax import compose, vmap_chunked

import jax
import jax.flatten_util
import jax.numpy as jnp

import numpy as np

from netket.stats import subtract_mean, sum
from netket.utils import mpi

from netket.utils.types import Array, Callable, PyTree, Scalar

from netket.jax import tree_cast, tree_conj, tree_axpy, tree_to_real


# TODO better name and move it somewhere sensible
def single_sample(forward_fn):
    """
    A decorator to make the forward_fn accept a single sample
    """

    def f(W, σ):
        return forward_fn(W, σ[jnp.newaxis, :])[0]

    return f


# TODO move it somewhere reasonable
def tree_subtract_mean(oks: PyTree) -> PyTree:
    """
    subtract the mean with MPI along axis 0 of every leaf
    """
    return jax.tree_map(partial(subtract_mean, axis=0), oks)  # MPI


def jacobian_real_holo(
    forward_fn: Callable, params: PyTree, samples: Array, chunk_size: int = None
) -> PyTree:
    """Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→R or holomorphic C→C, so single grad is enough

    Args:
        forward_fn: the log wavefunction ln Ψ
        params : a pytree of parameters p
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """

    def _jacobian_real_holo(forward_fn, params, samples):
        y, vjp_fun = jax.vjp(single_sample(forward_fn), params, samples)
        res, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
        return res

    return vmap_chunked(
        _jacobian_real_holo, in_axes=(None, None, 0), chunk_size=chunk_size
    )(forward_fn, params, samples)


def jacobian_cplx(
    forward_fn: Callable,
    params: PyTree,
    samples: Array,
    chunk_size: int = None,
    _build_fn: Callable = partial(jax.tree_multimap, jax.lax.complex),
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

    def _jacobian_cplx(forward_fn, params, samples, _build_fn):
        y, vjp_fun = jax.vjp(single_sample(forward_fn), params, samples)
        gr, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
        gi, _ = vjp_fun(np.array(-1.0j, dtype=jnp.result_type(y)))
        return _build_fn(gr, gi)

    return vmap_chunked(
        _jacobian_cplx, in_axes=(None, None, 0, None), chunk_size=chunk_size
    )(forward_fn, params, samples, _build_fn)


centered_jacobian_real_holo = compose(tree_subtract_mean, jacobian_real_holo)
centered_jacobian_cplx = compose(tree_subtract_mean, jacobian_cplx)


def _divide_by_sqrt_n_samp(oks, samples):
    """
    divide Oⱼₖ by √n
    """
    n_samp = samples.shape[0] * mpi.n_nodes  # MPI
    return jax.tree_map(lambda x: x / np.sqrt(n_samp), oks)


def _multiply_by_pdf(oks, pdf):
    """
    Computes  O'ⱼ̨ₖ = Oⱼₖ pⱼ .
    Used to multiply the log-derivatives by the probability density.
    """

    return jax.tree_map(
        lambda x: jax.lax.broadcast_in_dim(pdf, x.shape, (0,)) * x,
        oks,
    )


def stack_jacobian(centered_oks: PyTree) -> PyTree:
    """
    Return the real and imaginary parts of ΔOⱼₖ stacked along the sample axis
    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]
    """
    return jax.tree_map(
        lambda x: jnp.concatenate([x.real, x.imag], axis=0), centered_oks
    )


def stack_jacobian_tuple(centered_oks_re_im):
    """
    stack the real and imaginary parts of ΔOⱼₖ along the sample axis

    Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]

    Args:
        centered_oks_re_im : a tuple (ΔOᵣ, ΔOᵢ) of two PyTrees representing the real and imag part of ΔOⱼₖ
    """
    return jax.tree_multimap(
        lambda re, im: jnp.concatenate([re, im], axis=0), *centered_oks_re_im
    )


def _rescale(centered_oks):
    """
    compute ΔOₖ/√Sₖₖ and √Sₖₖ
    to do scale-invariant regularization (Becca & Sorella 2017, pp. 143)
    Sₖₗ/(√Sₖₖ√Sₗₗ) = ΔOₖᴴΔOₗ/(√Sₖₖ√Sₗₗ) = (ΔOₖ/√Sₖₖ)ᴴ(ΔOₗ/√Sₗₗ)
    """
    scale = jax.tree_map(
        lambda x: mpi.mpi_sum_jax(jnp.sum((x * x.conj()).real, axis=0, keepdims=True))[
            0
        ]
        ** 0.5,
        centered_oks,
    )
    centered_oks = jax.tree_multimap(jnp.divide, centered_oks, scale)
    scale = jax.tree_map(partial(jnp.squeeze, axis=0), scale)
    return centered_oks, scale


def _jvp(oks: PyTree, v: PyTree) -> Array:
    """
    Compute the matrix-vector product between the pytree jacobian oks and the pytree vector v
    """
    td = lambda x, y: jnp.tensordot(x, y, axes=y.ndim)
    return jax.tree_util.tree_reduce(jnp.add, jax.tree_multimap(td, oks, v))


def _vjp(oks: PyTree, w: Array) -> PyTree:
    """
    Compute the vector-matrix product between the vector w and the pytree jacobian oks
    """
    res = jax.tree_map(partial(jnp.tensordot, w, axes=1), oks)
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
        chunk_size: an int specfying the size of the chunks the gradient should be computed in (default: None)

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
        centered_jacobian_fun = centered_jacobian_real_holo
        jacobian_fun = jacobian_real_holo
    elif mode == "complex":
        split_complex_params = True  # convert C→C and R&C→C to R→C
        # centered_jacobian_fun = compose(stack_jacobian, centered_jacobian_cplx)

        # avoid converting to complex and then back
        # by passing around the oks as a tuple of two pytrees representing the real and imag parts
        centered_jacobian_fun = compose(
            stack_jacobian_tuple,
            partial(centered_jacobian_cplx, _build_fn=lambda *x: x),
        )
        jacobian_fun = jacobian_cplx
    elif mode == "holomorphic":
        split_complex_params = False
        centered_jacobian_fun = centered_jacobian_real_holo
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

        def f(W, σ):
            return forward_fn(reassemble(W), σ)

    else:
        f = forward_fn

    if pdf is None:
        centered_oks = _divide_by_sqrt_n_samp(
            centered_jacobian_fun(
                f,
                params,
                samples,
                chunk_size=chunk_size,
            ),
            samples,
        )
    else:
        oks = jacobian_fun(f, params, samples)
        oks_mean = jax.tree_map(partial(sum, axis=0), _multiply_by_pdf(oks, pdf))
        centered_oks = jax.tree_multimap(lambda x, y: x - y, oks, oks_mean)

        centered_oks = _multiply_by_pdf(centered_oks, jnp.sqrt(pdf))
    if rescale_shift:
        return _rescale(centered_oks)
    else:
        return centered_oks, None


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
