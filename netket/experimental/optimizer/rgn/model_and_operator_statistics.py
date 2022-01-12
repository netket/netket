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
from netket.stats import stats

import jax
import jax.flatten_util
import jax.numpy as jnp

import numpy as np

from netket.stats import subtract_mean, sum, mean
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


def tree_mean(oks: PyTree) -> PyTree:
    """
    subtract the mean with MPI along axis 0 of every leaf
    """
    return jax.tree_map(partial(mean, axis=0), oks)

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
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  

def jacobian_real_holo(
    forward_fn: Callable,
    params: PyTree,
    samples: Array,
    chunk_size: int = None,
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

    def _jacobian_real_holo(forward_fn, params, samples):
        y, vjp_fun = jax.vjp(single_sample(forward_fn), params, samples)
        res, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
        return res

    return vmap_chunked(
        _jacobian_real_holo, in_axes=(None, None, 0), chunk_size=chunk_size
    )(forward_fn, params, samples)


    return vmap_chunked(
        _jacobian_cplx, in_axes=(None, None, 0, None), chunk_size=chunk_size
    )(forward_fn, params, samples, _build_fn)

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

@partial(jax.jit, static_argnames=("forward_fn", "mode", "chunk_size"))
def centered_jacobian_and_mean(
    forward_fn: Callable,
    params: PyTree,
    samples: Array,
    mode: str,
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

    if mode == "real":
        split_complex_params = True  # convert C→R and R&C→R to R→R
        jacobian_fun = jacobian_real_holo
    elif mode == "complex":
        split_complex_params = True  # convert C→C and R&C→C to R→C
        # centered_jacobian_fun = compose(stack_jacobian, centered_jacobian_cplx)

        # avoid converting to complex and then back
        # by passing around the oks as a tuple of two pytrees representing the real and imag parts
        jacobian_fun = compose(
            stack_jacobian_tuple,
            partial(jacobian_cplx, _build_fn=lambda *x: x),
        )
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

        def f(W, σ):
            return forward_fn(reassemble(W), σ)

    else:
        f = forward_fn

    oks = _divide_by_sqrt_n_samp(
        jacobian_fun(
            f,
            params,
            samples,
            chunk_size=chunk_size,
        ),
        samples,
    )

    return tree_subtract_mean(oks), tree_mean(oks)


def en_and_rhessian_real_holo(
    forward_fn: Callable, params: PyTree, samples: Array, connected_samples: Array, matrix_elements: Array, chunk_size: int = None
) -> [Array, PyTree]:
    """Calculates the energy and one of the terms in the right hand side of the hessian

    Args:
        forward_fn: a function that generates the log wavefunction ln Ψ
        params : a pytree of parameters p 
        samples : an array of n samples σ of shape [n_samples,hilbert_size]
        connected_samples: an array of samples connected by the Hamiltonian of shape [n_samples,max_connected,hilbert_size]
        mels: matrix elements with respected to connected samples of shape [n_samples,max_connected] 

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """


    def _en_and_rhessian_real_holo(logpsi, params, σ, σp, mels):
        def E_loc(params):
            return mels*jnp.exp(logpsi(params,σp) - jax.lax.stop_gradient(single_sample(logpsi)(params,σ)))
        eloc, vjpfun = jax.vjp(E_loc, params)
        
        return eloc, vjp_fun(np.array(1.0, dtype=jnp.result_type(eloc)))

    return vmap_chunked(
        _en_and_rhessian_real_holo, in_axes=(None, None, 0,0,0), chunk_size=chunk_size
    )(f, params, samples,connected_samples,matrix_elements)

def en_and_rhessian_cplx(
    forward_fn: Callable, params: PyTree, samples: Array, connected_samples: Array, matrix_elements: Array, chunk_size: int = None,_build_fn: Callable = partial(jax.tree_multimap, jax.lax.complex),
) -> [Array, PyTree]:
    """Calculates the energy and one of the terms in the right hand side of the hessian

    Args:
        forward_fn: a function that generates the log wavefunction ln Ψ
        params : a pytree of parameters p 
        samples : an array of n samples σ of shape [n_samples,hilbert_size]
        connected_samples: an array of samples connected by the Hamiltonian of shape [n_samples,max_connected,hilbert_size]
        mels: matrix elements with respected to connected samples of shape [n_samples,max_connected] 

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    """


    def _en_and_rhessian_cplx(logpsi, params, σ, σp, mels):
        def E_loc(params):
            return mels*jnp.exp(logpsi(params,σp) - jax.lax.stop_gradient(single_sample(logpsi)(params,σ)))
        eloc, vjpfun = jax.vjp(E_loc, params)
        
        return eloc, _build_fn(vjp_fun(np.array(1.0, dtype=jnp.result_type(eloc))),vjp_fun(np.array(-1.0j, dtype=jnp.result_type(eloc))))

    return vmap_chunked(
        _en_and_rhessian_cplx, in_axes=(None, None, 0,0,0), chunk_size=chunk_size
    )(f, params, samples,connected_samples,matrix_elements)

def en_grad_and_rhessian(
    forward_fn: Callable, params: PyTree, samples: Array, connected_samples: Array, matrix_elements: Array,  mode: str, chunk_size: int = None,
) -> Tuple[Stats, PyTree, PyTree]:

    n_samples = σ.shape[0] * mpi.n_nodes

    if mode == "real" or "complex":
        # doesn't do anything if the params are already real
        params, reassemble = tree_to_real(params)

        def f(W, σ):
            return forward_fn(reassemble(W), σ)

    elif mode == "holomorphic":
        f = forward_fn
    else:
        raise NotImplementedError(
            'Differentiation mode should be one of "real", "complex", or "holomorphic", got {}'.format(
                mode
            )
        )

    if mode == "holomorphic" or "real": 
        eloc, rhessian = en_and_rhessian_real_holo(f,params,samples,connected_samples,matrix_elements,chunk_size)
    else:
        eloc, rhessian = en_and_rhessian_cplx(f,params,samples,connected_samples,matrix_elements,chunk_size, _build_fn=lambda x: stack_jacobian_tuple(x))


    rhessian = tree_subtract_mean(_divide_by_sqrt_n_samp(rhessian,samples))

    E = statistics(eloc.reshape(σ_shape[:-1]).T)

    eloc -= E.mean

    vjpfun = jax.vjp(forward_fn, params)

    grad = vjp_fun(jnp.conjugate(eloc)/n_samples)[0]

    return E, grad, rhessian 


def mat_vec(v: PyTree, oks: PyTree, rhes: PyTree, mean_grad: PyTree, eps: float, en: float, diag_shift: float) -> PyTree:
    """
    Compute ⟨O† O⟩v = ∑ₗ ⟨Oₖᴴ Oₗ⟩ vₗ
    """

    rhes = tree_axpy(1/eps-en,oks,rhes)

    res = tree_conj(_vjp(oks, _jvp(rhes, v).conjugate()))
    res2 = tree_conj(jax.tree_map(lambda x: _jvp(mean_grad, v)*x,v))

    res = jax.tree_multimap(lambda x,y,z: x - y + diag_shift*z,res,res2,v)

    return tree_cast(res, v)


