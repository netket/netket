# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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
from functools import partial

import numpy as _np
from jax import numpy as jnp
from jax.tree_util import (
    tree_flatten,
    tree_unflatten,
    tree_map,
    tree_multimap,
    tree_leaves,
)


def forward_scalar(pars, forward_fn, x):
    """
    Performs the forward pass of a neural network on a single state (vector) input.
    """
    return forward_fn(pars, jnp.expand_dims(x, 0)).reshape(())


forward_apply = jax.jit(
    lambda pars, forward_fn, x: forward_fn(pars, x), static_argnums=1
)

##
@jax.jit
def tree_size(tree):
    """
    Returns the sum of the size of all leaves in the tree.
    It's equivalent to the number of scalars in the pytree.
    """
    return sum(tree_leaves(tree_map(lambda x: x.size, tree)))


def tree_leaf_iscomplex(pars):
    """
    Returns true if at least one leaf in the tree has complex dtype.
    """

    def _has_complex_dtype(x):
        #  Returns true if x is complex
        return jnp.issubdtype(x.dtype, jnp.complexfloating)

    return any(jax.tree_leaves(jax.tree_map(_has_complex_dtype, pars)))


def outdtype(forward_fn, pars, v):
    """
    Returns the dtype of forward_fn(pars, v)
    """
    if v.ndim > 1:
        v = v.reshape(-1, v.shape[-1])[0, :]
    return forward_scalar(pars, forward_fn, v).dtype


def outdtype_iscomplex(forward_fn, pars, v):
    """
    Returns true if forward_fn(pars, v) is complex
    """
    return jnp.issubdtype(outdtype(forward_fn, pars, v), jnp.complexfloating)


# _grad_CC, _RR and _RC are the batched gradient functions for machines going
# from R -> C, R->R and R->C. Ditto for vjp
# Thee reason why R->C is more complicated is that it splits the calculation
# into the real and complex part in order to be more efficient.

_grad_CC = jax.grad(forward_scalar, holomorphic=True)
_grad_RR = jax.grad(forward_scalar)


def _grad_RC(pars, forward_fn, v):
    grad_r = jax.grad(
        lambda pars, forward_fn, v: forward_scalar(pars, forward_fn, v).real
    )(pars, forward_fn, v)
    grad_j = jax.grad(
        lambda pars, forward_fn, v: forward_scalar(pars, forward_fn, v).imag
    )(pars, forward_fn, v)

    r_flat, r_fun = tree_flatten(grad_r)
    j_flat, j_fun = tree_flatten(grad_j)

    grad_flat = [re + 1j * im for re, im in zip(r_flat, j_flat)]
    return tree_unflatten(r_fun, grad_flat)


def _grad(pars, forward_fn, v):
    if tree_leaf_iscomplex(pars):
        if outdtype_iscomplex(forward_fn, pars, v):  # C -> C
            return _grad_CC(pars, forward_fn, v)
        else:
            raise RuntimeError("C->R function detected, but not supported.")
    else:
        if outdtype_iscomplex(forward_fn, pars, v):  # R -> C
            return _grad_RC(pars, forward_fn, v)
        else:
            return _grad_RR(pars, forward_fn, v)


grad = jax.jit(jax.vmap(_grad, in_axes=(None, None, 0)), static_argnums=1)


def _vjp_CC(pars, forward_fn, v, vec, conjugate):
    vals, f_vjp = jax.vjp(forward_fn, pars, v.reshape((-1, v.shape[-1])))

    out = f_vjp(vec.reshape(vals.shape).conjugate())[0]

    if conjugate:
        out = tree_map(jnp.conjugate, out)

    return out


def _vjp_RR(pars, forward_fn, v, vec, conjugate):
    vals, f_jvp = jax.vjp(forward_fn, pars, v.reshape((-1, v.shape[-1])))

    out_r = f_jvp(vec.reshape(vals.shape).real)[0]
    out_i = f_jvp(-vec.reshape(vals.shape).imag)[0]

    if conjugate:
        out = tree_multimap(lambda re, im: re - 1j * im, out_r, out_i)
    else:
        out = tree_multimap(lambda re, im: re + 1j * im, out_r, out_i)

    return out


def _vjp_RC(pars, forward_fn, v, vec, conjugate):
    v = v.reshape((-1, v.shape[-1]))
    vals_r, f_jvp_r = jax.vjp(lambda pars, v: forward_fn(pars, v).real, pars, v)

    vals_j, f_jvp_j = jax.vjp(lambda pars, v: forward_fn(pars, v).imag, pars, v)

    vec_r = vec.reshape(vals_r.shape).real
    vec_j = vec.reshape(vals_r.shape).imag

    # val = vals_r + vals_j
    vr_jr = f_jvp_r(vec_r)[0]
    vj_jr = f_jvp_r(vec_j)[0]
    vr_jj = f_jvp_j(vec_r)[0]
    vj_jj = f_jvp_j(vec_j)[0]

    r = tree_multimap(
        lambda re, im: re - 1j * im,
        vr_jr,
        vj_jr,
    )
    i = tree_multimap(lambda re, im: re - 1j * im, vr_jj, vj_jj)
    out = tree_multimap(lambda re, im: re + 1j * im, r, i)

    if conjugate:
        out = tree_map(jnp.conjugate, out)

    return out


#  This function dispatches to the right
def _vjp(pars, forward_fn, v, vec, conjugate):

    # output dtype
    out_dtype = forward_scalar(pars, forward_fn, v[0, :]).dtype
    out_dtype = _np.promote_types(out_dtype, vec.dtype)
    # convert the sensitivity to right dtype
    vec = jnp.asarray(vec, dtype=out_dtype)

    if tree_leaf_iscomplex(pars):
        if jnp.issubdtype(out_dtype, jnp.complexfloating):  # C -> C
            return _vjp_CC(pars, forward_fn, v, vec, conjugate)
        elif jnp.issubdtype(out_dtype, jnp.floating):  # C -> R
            raise RuntimeError("C->R function detected, but not supported.")
    else:
        if jnp.issubdtype(out_dtype, jnp.complexfloating):  # R -> C
            return _vjp_RC(pars, forward_fn, v, vec, conjugate)
        elif jnp.issubdtype(out_dtype, jnp.floating):  # R -> R
            return _vjp_RR(pars, forward_fn, v, vec, conjugate)


vjp = jax.jit(_vjp, static_argnums=(1, 4))
