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

# test qgt_onthefly_logic with inhomogeneous parameters

import pytest

import jax
import jax.numpy as jnp
import jax.flatten_util
from netket.optimizer.qgt import qgt_onthefly_logic, qgt_jacobian_pytree_logic

from functools import partial
import itertools

from netket import jax as nkjax

from .. import common

pytestmark = common.skipif_mpi

# TODO move the transformation and tree utils out of the test


# transform inhomogeneous parameters into a flattened array of reals parameters
# without needing to pad


def flatten(x):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    return x_flat


def toreal(x):
    if jnp.iscomplexobj(x):
        return jnp.array([x.real, x.imag])
    else:
        return x


def tree_toreal(x):
    return jax.tree_map(toreal, x)


def tree_toreal_flat(x):
    return flatten(tree_toreal(x))


# invert the transformation tree_toreal_flat using linear_transpose (AD)
def reassemble_complex(x, target, fun=tree_toreal_flat):
    # target: a tree with the expected shape and types of the result
    (res,) = jax.linear_transpose(fun, target)(x)
    res = qgt_onthefly_logic.tree_conj(res)
    # fix the dtypes:
    return nkjax.tree_cast(res, target)


def tree_allclose(t1, t2):
    t = jax.tree_multimap(jnp.allclose, t1, t2)
    return all(jax.tree_util.tree_flatten(t)[0])


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_multimap(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


class Example:
    def f_real_flat(self, p, samples):
        return self.f(reassemble_complex(p, target=self.params), samples)

    def f_real_flat_scalar(self, params, x):
        return self.f_real_flat(params, jnp.expand_dims(x, 0))[0]

    # same as in nk.machine.Jax R->C
    @partial(jax.vmap, in_axes=(None, None, 0))
    def grads_real(self, params, x):
        r = jax.grad(lambda pars, v: self.f_real_flat_scalar(pars, v).real)(params, x)
        i = jax.grad(lambda pars, v: self.f_real_flat_scalar(pars, v).imag)(params, x)
        return jax.lax.complex(r, i)

    @property
    def n_samp(self):
        return len(self.samples)

    def __init__(self, n_samp, seed, outdtype, pardtype, holomorphic):

        self.dtype = outdtype

        self.target = {
            "a": jnp.array([[[0j], [0j]]], dtype=jnp.complex128),
            "b": jnp.array(0, dtype=jnp.float64),
            "c": jnp.array(0j, dtype=jnp.complex64),
        }

        if pardtype is None:  # mixed precision as above
            pass
        else:
            self.target = jax.tree_map(lambda x: x.astype(pardtype), self.target)

        k = jax.random.PRNGKey(seed)
        k1, k2, k3, k4, k5 = jax.random.split(k, 5)

        self.samples = jax.random.normal(k1, (n_samp, 2))
        self.w = jax.random.normal(k2, (n_samp,), self.dtype).astype(
            self.dtype
        )  # TODO remove astype once its fixed in jax
        self.params = tree_random_normal_like(k3, self.target)
        self.v = tree_random_normal_like(k4, self.target)
        self.grad = tree_random_normal_like(k5, self.target)

        if holomorphic:

            @partial(jax.vmap, in_axes=(None, 0))
            def f(params, x):
                return (
                    params["a"][0][0][0] * x[0]
                    + params["b"] * x[1]
                    + params["c"] * (x[0] * x[1])
                    + jnp.sin(x[1] * params["a"][0][1][0])
                    * jnp.cos(x[0] * params["b"] + 1j)
                    * params["c"]
                ).astype(self.dtype)

        else:

            @partial(jax.vmap, in_axes=(None, 0))
            def f(params, x):
                return (
                    params["a"][0][0][0].conjugate() * x[0]
                    + params["b"] * x[1]
                    + params["c"] * (x[0] * x[1])
                    + jnp.sin(x[1] * params["a"][0][1][0])
                    * jnp.cos(x[0] * params["b"].conjugate() + 1j)
                    * params["c"].conjugate()
                ).astype(self.dtype)

        self.f = f

        self.params_real_flat = tree_toreal_flat(self.params)
        self.grad_real_flat = tree_toreal_flat(self.grad)
        self.v_real_flat = tree_toreal_flat(self.v)
        self.ok_real = self.grads_real(self.params_real_flat, self.samples)
        self.okmean_real = self.ok_real.mean(axis=0)
        self.dok_real = self.ok_real - self.okmean_real
        self.S_real = (
            self.dok_real.conjugate().transpose() @ self.dok_real / n_samp
        ).real
        self.scale = jnp.sqrt(self.S_real.diagonal())
        self.S_real_scaled = self.S_real / (jnp.outer(self.scale, self.scale))


@pytest.fixture
def e(n_samp, outdtype, pardtype, holomorphic, seed=123):
    return Example(n_samp, seed, outdtype, pardtype, holomorphic)


rt = [jnp.float32, jnp.float64]
ct = [jnp.complex64, jnp.complex128]
nct = [None] + ct  # None means inhomogeneous
r_r_test_types = list(itertools.product(rt, rt))
c_c_test_types = list(itertools.product(ct, ct))
r_c_test_types = list(itertools.product(ct, rt))
rc_c_test_types = list(itertools.product(ct, [None]))
c_r_test_types = list(itertools.product(rt, ct))
rc_r_test_types = list(itertools.product(rt, [None]))

test_types = r_r_test_types + c_c_test_types + r_c_test_types + rc_c_test_types
all_test_types = test_types + c_r_test_types + rc_r_test_types

# tests


@pytest.mark.parametrize("holomorphic", [True])
@pytest.mark.parametrize("n_samp", [0])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_reassemble_complex(e):
    assert tree_allclose(
        e.params, reassemble_complex(tree_toreal_flat(e.params), target=e.target)
    )


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [24, 1024])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("outdtype, pardtype", all_test_types)
@pytest.mark.parametrize("chunk_size", [8, None])
def test_matvec(e, jit, chunk_size):
    diag_shift = 0.01

    def f(params_model_state, x):
        return e.f(params_model_state["params"], x)

    if chunk_size is None:
        mat_vec_factory = qgt_onthefly_logic.mat_vec_factory
        samples = e.samples
    else:
        mat_vec_factory = qgt_onthefly_logic.mat_vec_chunked_factory
        samples = e.samples.reshape((-1, chunk_size) + e.samples.shape[1:])

    mv = mat_vec_factory(f, e.params, {}, samples)
    if jit:
        mv = jax.jit(mv)
    actual = mv(e.v, diag_shift)
    expected = reassemble_complex(
        e.S_real @ e.v_real_flat + diag_shift * e.v_real_flat, target=e.target
    )
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [24, 1024])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("outdtype, pardtype", all_test_types)
@pytest.mark.parametrize("chunk_size", [8, None])
def test_matvec_linear_transpose(e, jit, chunk_size):
    def f(params_model_state, x):
        return e.f(params_model_state["params"], x)

    if chunk_size is None:
        mat_vec_factory = qgt_onthefly_logic.mat_vec_factory
        samples = e.samples
    else:
        mat_vec_factory = qgt_onthefly_logic.mat_vec_chunked_factory
        samples = e.samples.reshape((-1, chunk_size) + e.samples.shape[1:])

    mv = mat_vec_factory(f, e.params, {}, samples)

    def mvt(v, w):
        (res,) = jax.linear_transpose(lambda v_: mv(v_, 0.0), v)(w)
        return res

    if jit:
        mv = jax.jit(mv)
        mvt = jax.jit(mvt)

    w = e.v
    actual = mvt(e.v, w)

    # use that S is hermitian:
    # S^T = (O^H O)^T = O^T O* = (O^H O)* = S*
    # S^T w = S* w = (S w*)*
    expected = nkjax.tree_conj(
        mv(
            nkjax.tree_conj(w),
            0.0,
        )
    )
    # (expected,) = jax.linear_transpose(lambda v_: reassemble_complex(S_real @ tree_toreal_flat(v_), target=e.target), v)(v)
    assert tree_allclose(actual, expected)


# TODO separate test for prepare_centered_oks
@pytest.mark.parametrize("holomorphic", [True])
@pytest.mark.parametrize("n_samp", [25, 1024])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("chunk_size", [7, None])
@pytest.mark.parametrize(
    "outdtype, pardtype", r_r_test_types + c_c_test_types + r_c_test_types
)
def test_matvec_treemv(e, jit, holomorphic, pardtype, outdtype, chunk_size):
    mv = qgt_jacobian_pytree_logic._mat_vec

    if not nkjax.is_complex_dtype(pardtype) and nkjax.is_complex_dtype(outdtype):
        centered_jacobian_fun = qgt_jacobian_pytree_logic.centered_jacobian_cplx
    else:
        centered_jacobian_fun = qgt_jacobian_pytree_logic.centered_jacobian_real_holo
    centered_jacobian_fun = partial(centered_jacobian_fun, chunk_size=chunk_size)
    if jit:
        mv = jax.jit(mv)
        centered_jacobian_fun = jax.jit(centered_jacobian_fun, static_argnums=0)

    centered_oks = centered_jacobian_fun(e.f, e.params, e.samples)
    centered_oks = qgt_jacobian_pytree_logic._divide_by_sqrt_n_samp(
        centered_oks, e.samples
    )
    actual = mv(e.v, centered_oks)
    expected = reassemble_complex(e.S_real @ e.v_real_flat, target=e.target)
    assert tree_allclose(actual, expected)


# TODO separate test for prepare_centered_oks
# TODO test C->R ?
@pytest.mark.parametrize("holomorphic", [True, False])
@pytest.mark.parametrize("n_samp", [25, 1024])
@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_matvec_treemv_modes(e, jit, holomorphic, pardtype, outdtype):
    diag_shift = 0.01
    model_state = {}
    rescale_shift = False

    def apply_fun(params, samples):
        return e.f(params["params"], samples)

    mv = qgt_jacobian_pytree_logic.mat_vec

    homogeneous = pardtype is not None

    if not nkjax.is_complex_dtype(outdtype):
        mode = "real"
    elif homogeneous and nkjax.is_complex_dtype(pardtype) and holomorphic:
        mode = "holomorphic"
    else:
        mode = "complex"

    if mode == "holomorphic":
        v = e.v
        reassemble = lambda x: x
    else:
        v, reassemble = nkjax.tree_to_real(e.v)

    if jit:
        mv = jax.jit(mv)

    centered_oks, _ = qgt_jacobian_pytree_logic.prepare_centered_oks(
        apply_fun, e.params, e.samples, model_state, mode, rescale_shift
    )
    actual = reassemble(mv(v, centered_oks, diag_shift))
    expected = reassemble_complex(
        e.S_real @ e.v_real_flat + diag_shift * e.v_real_flat, target=e.target
    )
    assert tree_allclose(actual, expected)


@pytest.mark.parametrize("holomorphic", [True])
@pytest.mark.parametrize("n_samp", [25, 1024])
@pytest.mark.parametrize(
    "outdtype, pardtype", r_r_test_types + c_c_test_types + r_c_test_types
)
def test_scale_invariant_regularization(e, outdtype, pardtype):

    if not nkjax.is_complex_dtype(pardtype) and nkjax.is_complex_dtype(outdtype):
        centered_jacobian_fun = qgt_jacobian_pytree_logic.centered_jacobian_cplx
    else:
        centered_jacobian_fun = qgt_jacobian_pytree_logic.centered_jacobian_real_holo

    mv = qgt_jacobian_pytree_logic._mat_vec
    centered_oks = centered_jacobian_fun(e.f, e.params, e.samples)
    centered_oks = qgt_jacobian_pytree_logic._divide_by_sqrt_n_samp(
        centered_oks, e.samples
    )

    centered_oks_scaled, scale = qgt_jacobian_pytree_logic._rescale(centered_oks)
    actual = mv(e.v, centered_oks_scaled)
    expected = reassemble_complex(e.S_real_scaled @ e.v_real_flat, target=e.target)
    assert tree_allclose(actual, expected)


# TODO test with MPI
