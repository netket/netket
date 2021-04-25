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

from .sr_onthefly_logic import tree_cast, tree_conj, tree_axpy


# TODO cheapest way to calculate the gradients?


@partial(jax.vmap, in_axes=(None, None, 0))
def perex_grads_rr_cc(forward_fn, params, samples):
    def f(p, x):
        return forward_fn(p, jnp.expand_dims(x, 0))[0]

    y, vjp_fun = jax.vjp(f, params, samples)
    res, _ = vjp_fun(np.ones((), dtype=jnp.result_type(y)))
    return res


@partial(jax.vmap, in_axes=(None, None, 0))
def perex_grads_rc(forward_fn, params, samples):
    fr = lambda p, x: forward_fn(p, jnp.expand_dims(x, 0))[0].real
    fi = lambda p, x: forward_fn(p, jnp.expand_dims(x, 0))[0].imag
    yr, vjp_funr = jax.vjp(fr, params, samples)
    yi, vjp_funi = jax.vjp(fi, params, samples)
    gr, _ = vjp_funr(np.ones((), dtype=jnp.result_type(yr)))
    gi, _ = vjp_funi(np.ones((), dtype=jnp.result_type(yi)))
    return jax.tree_multimap(jax.lax.complex, gr, gi)


def perex_grads(forward_fn, params, samples):
    o = jax.eval_shape(forward_fn, params, samples)
    if not nkjax.tree_leaf_iscomplex(params) and nkjax.is_complex(o):
        return perex_grads_rc(forward_fn, params, samples)
    else:
        return perex_grads_rr_cc(forward_fn, params, samples)
    # TODO inhomogeneous R&C -> C


def sub_mean(oks):
    return jax.tree_map(partial(subtract_mean, axis=0), oks)  # MPI


def prepare_doks(forward_fn, params, samples):
    oks = perex_grads(forward_fn, params, samples)
    n_samp = samples.shape[0] * n_nodes  # MPI
    # TODO where to divide by n_samp?
    oks = jax.tree_map(lambda x: x / np.sqrt(n_samp), oks)
    return sub_mean(oks)


def jvp(oks, v):
    td = lambda x, y: jnp.tensordot(x, y, axes=y.ndim)
    return jax.tree_util.tree_reduce(jnp.add, jax.tree_multimap(td, oks, v))


def _cast(x, target):
    res = x if jnp.iscomplexobj(target) else x.real
    res = res.astype(target.dtype)
    return res


def vjp(oks, w):
    res = jax.tree_map(partial(jnp.tensordot, w, axes=1), oks)
    return jax.tree_map(sum_inplace, res)  # MPI


def _mat_vec(v, oks):
    res = tree_conj(vjp(oks, jvp(oks, v).conjugate()))
    return tree_cast(res, v)


def mat_vec(v, oks, diag_shift):
    return tree_axpy(diag_shift, v, _mat_vec(v, oks))
