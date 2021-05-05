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
import jax.flatten_util
import jax.numpy as jnp

import numpy as np
from functools import partial

from netket.stats import sum_inplace, subtract_mean
from netket.utils import n_nodes
import netket.jax as nkjax

from netket.utils.types import Array, Callable, PyTree, Scalar

from netket.jax import tree_cast, tree_conj, tree_axpy, tree_to_real

def sub_mean(oks: PyTree) -> PyTree:
    return jax.tree_map(partial(subtract_mean, axis=0), oks)  # MPI

@partial(jax.vmap, in_axes=(None, None, 0))
def vmap_grad_real_holo(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    '''Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→R or holomorphic C→C, so single grad is enough

    Args:
        forward_fn: the log wavefunction, partialed with any model state, limited to single sample, taking a split real-imaginary PyTree in R→R mode
        params : a pytree of parameters p, real-imaginary split in R→R mode
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree
    '''
    y, vjp_fun = jax.vjp(forward_fn, params, samples)
    res, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    return res

def vmap_grad_centered_real_holo(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    '''Calculates centred Jacobian (i.e., subtracts MPI mean from vmap_grad)'''
    return sub_mean(vmap_grad_rr_cc(forward_fn, params, samples))


@partial(jax.vmap, in_axes=(None, None, 0))
def vmap_grad_cplx(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    '''Calculates Jacobian entries by vmapping grad.
    Assumes the function is R→C, backpropagates 1 and -1j

    Args:
        forward_fn: the log wavefunction, partialed with any model state, limited to single sample, taking a split real-imaginary PyTree
        params : a pytree of parameters p, real-imaginary split
        samples : an array of n samples σ

    Returns:
        The Jacobian matrix ∂/∂pₖ ln Ψ(σⱼ) as a PyTree'''
    y, vjp_fun = jax.vjp(forward_fn, params, samples)
    gr, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    gi, _ = vjp_fun(np.array(-1.0j, dtype=jnp.result_type(y)))
    return gr, gi

def vmap_grad_centered_cplx(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    '''Calculates centred Jacobian (i.e., subtracts MPI mean from vmap_grad)'''
    gr, gi = vmap_grad_rc(forward_fn, params, samples)
    # Return the real and imaginary parts of ΔOⱼₖ stacked along the sample axis
    # Re[S] = Re[(ΔOᵣ + i ΔOᵢ)ᴴ(ΔOᵣ + i ΔOᵢ)] = ΔOᵣᵀ ΔOᵣ + ΔOᵢᵀ ΔOᵢ = [ΔOᵣ ΔOᵢ]ᵀ [ΔOᵣ ΔOᵢ]
    return jax.tree_multimap(lambda re, im: jnp.concatenate([re, im], axis=0), sub_mean(gr), sub_mean(gi))


def vmap_grad_centered(forward_fn: Callable, params: PyTree, samples: Array, model_state: Optional[PyTree], mode: str) -> PyTree:
    """
    compute the jacobian of forward_fn(params, samples) w.r.t params
    as a pytree using vmapped gradients for efficiency

    mode can be 'real', 'complex', 'holomorphic'
    TODO: add support for splitting complex pytree leaves into real and imag parts so there be a good default mode
    """
    if mode == "real":
        # Apply real-imaginary split
        params, reassemble = tree_to_real(params)
        # Preapply model state and reassemble to forward_fn, restrict to one sample
        def f(W, σ):
            return forward_fn({"params": reassemble(W), **model_state}, σ[jnp.newaxis, :])[0]
        return vmap_grad_centered_real_holo(f, params, samples)
    elif mode == "complex":
        # Apply real-imaginary split
        params, reassemble = tree_to_real(params)
        # Preapply model state and reassemble to forward_fn, restrict to one sample
        def f(W, σ):
            return forward_fn({"params": reassemble(W), **model_state}, σ[jnp.newaxis, :])[0]
        return vmap_grad_centered_cplx(f, params, samples)
    elif mode == "holomorphic":
        # Preapply model state and reassemble to forward_fn, restrict to one sample
        def f(W, σ):
            return forward_fn({"params": W, **model_state}, σ[jnp.newaxis, :])[0]
        return vmap_grad_centered_real_holo(f, params, samples)
    else:
        raise RuntimeError # invalid modes should be caught earlier
    

@partial(jax.jit, static_argnums=(0,4,5))
def prepare_doks(forward_fn: Callable, params: PyTree, samples: Array, model_state: Optional[PyTree], mode: str, rescale_shift: bool) -> PyTree:
    """
    compute ΔOⱼₖ = Oⱼₖ - ⟨Oₖ⟩ = ∂/∂pₖ ln Ψ(σⱼ) - ⟨∂/∂pₖ ln Ψ⟩
    divided by √n

    Args:
        forward_fn: the vectorised log wavefunction  ln Ψ(p; σ)
        params : a pytree of parameters p
        samples : an array of n samples σ
        model_state: untrained state parameters of the model
        mode: differentiation mode, must be one of 'R2R', 'R2C', 'holomorphic'
        rescale_shift: whether scale-invariant regularisation should be used (default: True)

    Returns:
        if not rescale_shift:
            a pytree representing the centered jacobian of ln Ψ evaluated at the samples σ, divided by √n; 
            None
        else:
            the same pytree, but the entries for each parameter normalised to unit norm; 
            pytree containing the norms that were divided out (same shape as params)

    """
    doks = vmap_grad_centered(forward_fn, params, samples, model_state, mode)
    n_samp = samples.shape[0] * n_nodes  # MPI
    doks = jax.tree_map(lambda x: x / np.sqrt(n_samp), doks)

    if rescale_shift:
        scale = jax.tree_map(lambda x: sum_inplace(jnp.sum((x * x.conj()).real, axis=0, keepdims=True))**0.5, doks)
        doks = jax.tree_multimap(jnp.divide, doks, scale)
        scale = jax.tree_map(partial(jnp.squeeze, axis=0), scale)
        return doks, scale
    else:
        return doks, None


def jvp(oks: PyTree, v: PyTree) -> Array:
    """
    Compute the matrix-vector product between the pytree jacobian oks and the pytree vector v
    In R→R and R→C modes, v must be real-imaginary split
    """
    td = lambda x, y: jnp.tensordot(x, y, axes=y.ndim)
    return jax.tree_util.tree_reduce(jnp.add, jax.tree_multimap(td, oks, v))


def vjp(oks: PyTree, w: Array) -> PyTree:
    """
    Compute the vector-matrix product between the vector w and the pytree jacobian oks
    In R→R and R→C modes, the output is real-imaginary split
    """
    res = jax.tree_map(partial(jnp.tensordot, w, axes=1), oks)
    return jax.tree_map(sum_inplace, res)  # MPI


def _mat_vec(v: PyTree, oks: PyTree) -> PyTree:
    """
    Compute S v = 1/n ⟨ΔO† ΔO⟩v = 1/n ∑ₗ ⟨ΔOₖᴴ ΔOₗ⟩ vₗ
    In R→R and R→C modes, the v and the output are real-imaginary split
    """
    res = tree_conj(vjp(oks, jvp(oks, v).conjugate()))
    return tree_cast(res, v)


def mat_vec(v: PyTree, oks: PyTree, diag_shift: Scalar) -> PyTree:
    """
    Compute (S + δ) v = 1/n ⟨ΔO† ΔO⟩v + δ v = ∑ₗ 1/n ⟨ΔOₖᴴΔOₗ⟩ vₗ + δ vₗ
    In R→R and R→C modes, the v and the output are real-imaginary split

    Args:
        v: pytree representing the vector v
        oks: pytree of gradients 1/√n ΔOⱼₖ
        diag_shift: a scalar diagonal shift δ
    Returns:
        a pytree corresponding to the sr matrix-vector product (S + δ) v
    """
    return tree_axpy(diag_shift, v, _mat_vec(v, oks))
