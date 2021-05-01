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

from .sr_onthefly_logic import tree_cast, tree_conj, tree_axpy


# TODO cheapest way to calculate the gradients?


@partial(jax.vmap, in_axes=(None, None, 0))
def vmap_grad_rr_cc(forward_fn, params, samples):
    def f(p, x):
        return forward_fn(p, jnp.expand_dims(x, 0))[0]

    y, vjp_fun = jax.vjp(f, params, samples)
    res, _ = vjp_fun(np.ones((), dtype=jnp.result_type(y)))
    return res


@partial(jax.vmap, in_axes=(None, None, 0))
def vmap_grad_rc(forward_fn, params, samples):
    def f(p, x):
        return forward_fn(p, jnp.expand_dims(x, 0))[0]

    y, vjp_fun = jax.vjp(f, params, samples)
    gr, _ = vjp_fun(np.array(1.0, dtype=jnp.result_type(y)))
    gi, _ = vjp_fun(np.array(-1.0j, dtype=jnp.result_type(y)))
    return jax.tree_multimap(jax.lax.complex, gr, gi)


def vmap_grad(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    """
    compute the jacobian of forward_fn(params, samples) w.r.t params
    as a pytree using vmapped gradients for efficiency
    """
    complex_output = nkjax.is_complex(jax.eval_shape(forward_fn, params, samples))
    real_params = not nkjax.tree_leaf_iscomplex(params)
    if real_params and complex_output:
        return vmap_grad_rc(forward_fn, params, samples)
    else:
        return vmap_grad_rr_cc(forward_fn, params, samples)


def sub_mean(oks: PyTree) -> PyTree:
    return jax.tree_map(partial(subtract_mean, axis=0), oks)  # MPI


def prepare_doks(forward_fn: Callable, params: PyTree, samples: Array) -> PyTree:
    """
    compute ΔOⱼₖ = Oⱼₖ - ⟨Oₖ⟩ = ∂/∂pₖ ln Ψ(σⱼ) - ⟨∂/∂pₖ ln Ψ⟩
    divided by √n

    Args:
        forward_fn: the vectorised log wavefunction  ln Ψ(p; σ)
        params : a pytree of parameters p
        samples : an array of n samples σ

    Returns:
        a pytree representing the centered jacobian of ln Ψ evaluated at the samples σ
        divided by √n
    """
    oks = vmap_grad(forward_fn, params, samples)
    n_samp = samples.shape[0] * n_nodes  # MPI
    oks = jax.tree_map(lambda x: x / np.sqrt(n_samp), oks)
    doks = sub_mean(oks)  # MPI

    real_params = not nkjax.tree_leaf_iscomplex(params)
    complex_oks = nkjax.tree_leaf_iscomplex(oks)
    if real_params and complex_oks:
        # R->C
        # convert the complex oks to real ones with twice the number of rows
        # Re[S] = Re[(Oᵣ + i Oᵢ)ᴴ(Oᵣ + i Oᵢ)] = Oᵣᵀ Oᵣ + Oᵢᵀ Oᵢ = [Oᵣ Oᵢ] [Oᵣ Oᵢ]ᵀ
        doks = jax.tree_map(lambda x: jnp.concatenate([x.real, x.imag], axis=0), doks)

    return doks


def jvp(oks: PyTree, v: PyTree) -> Array:
    """
    compute the matrix-vector product between the pytree jacobian oks and the pytree vector v
    """
    td = lambda x, y: jnp.tensordot(x, y, axes=y.ndim)
    return jax.tree_util.tree_reduce(jnp.add, jax.tree_multimap(td, oks, v))


def vjp(oks: PyTree, w: Array) -> PyTree:
    """
    compute the vector-matrix product between the vector w and the pytree jacobian oks
    """
    res = jax.tree_map(partial(jnp.tensordot, w, axes=1), oks)
    return jax.tree_map(sum_inplace, res)  # MPI


def _mat_vec(v: PyTree, oks: PyTree) -> PyTree:
    """
    compute S v = 1/n ⟨O†O⟩v = 1/n ∑ₗ ⟨OₖᴴOₗ⟩ vₗ
    """
    res = tree_conj(vjp(oks, jvp(oks, v).conjugate()))
    return tree_cast(res, v)


def mat_vec(v: PyTree, oks: PyTree, diag_shift: Scalar) -> PyTree:
    """
    compute (S + δ) v = 1/n ⟨O†O⟩v + δ v = ∑ₗ 1/n ⟨OₖᴴOₗ⟩ vₗ + δ vₗ

    Args:
        v: pytree representing the vector v
        oks: pytree of gradients 1/√n Oⱼₖ or 1/√n ΔOⱼₖ
        diag_shift: a scalar diagonal shift δ
    Returns:
        a pytree corresponding to the sr matrix-vector product (S + δ) v
    """
    return tree_axpy(diag_shift, v, _mat_vec(v, oks))
