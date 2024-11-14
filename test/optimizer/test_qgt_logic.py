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
from functools import partial

import math
import numpy as np

import jax
import jax.numpy as jnp
import jax.flatten_util
from jax.tree_util import Partial

import itertools

from netket import jax as nkjax
from netket import stats as nkstats
from netket.utils import mpi
from netket.optimizer.qgt import (
    qgt_onthefly_logic,
    qgt_jacobian_pytree,
    qgt_jacobian_common,
)
import netket as nk
from netket.jax.sharding import distribute_to_devices_along_axis, device_count_per_rank

from .. import common

pytestmark = common.skipif_distributed

# TODO move the transformation and tree utils out of the test


# transform inhomogeneous parameters into a flattened array of real parameters
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
    return jax.tree_util.tree_map(toreal, x)


def tree_toreal_flat(x):
    return flatten(tree_toreal(x))


# invert the transformation tree_toreal_flat using linear_transpose (AD)
def reassemble_complex(x, target, fun=tree_toreal_flat):
    # target: a tree with the expected shape and types of the result
    (res,) = jax.linear_transpose(fun, target)(x)
    res = qgt_onthefly_logic.tree_conj(res)
    # fix the dtypes:
    return nkjax.tree_cast(res, target)


def assert_tree_allclose(t1, t2, rtol=None, atol=0):
    t1_s = jax.tree_util.tree_structure(t1)
    t2_s = jax.tree_util.tree_structure(t1)
    assert t1_s == t2_s

    def assert_allclose(x, y, rtol, atol):
        if rtol is None:
            if x.dtype == np.dtype("float32"):
                rtol = 1e-5
            elif x.dtype == np.dtype("complex64"):
                rtol = 1e-5
            else:
                rtol = 1e-6
        np.testing.assert_allclose(x, y, rtol, atol)

    jax.tree_util.tree_map(
        lambda x, y: assert_allclose(x, y, rtol=rtol, atol=atol), t1, t2
    )


def tree_samedtypes(t1, t2):
    def _same_dtypes(x, y):
        assert x.dtype == y.dtype
        assert x.weak_type == y.weak_type

    jax.tree_util.tree_map(_same_dtypes, t1, t2)


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


def astype_unsafe(x, dtype):
    """
    this function is equivalent to x.astype(dtype) but
    does not raise a complexwarning, which we treat as an error
    in our tests
    """
    if not nkjax.is_complex_dtype(dtype):
        x = x.real
    return x.astype(dtype)


def tree_subtract_mean(tree):
    return jax.tree_util.tree_map(lambda x: nkstats.subtract_mean(x, axis=0), tree)


def divide_by_sqrt_n_samp(oks, samples):
    n_samp = samples.shape[0] * mpi.n_nodes  # MPI
    sqrt_n = math.sqrt(n_samp)  # enforce weak type
    return jax.tree_util.tree_map(lambda x: x / sqrt_n, oks)


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

    def __init__(self, n_samp, seed, outdtype, pardtype, holomorphic, offset=0.0):
        self.dtype = outdtype

        self.target = {
            "a": jnp.array([[[0j], [0j]]], dtype=jnp.complex128),
            "b": jnp.array(0, dtype=jnp.float64),
            "c": jnp.array(0j, dtype=jnp.complex64),
        }

        if pardtype is None:  # mixed precision as above
            pass
        else:
            self.target = jax.tree_util.tree_map(
                lambda x: astype_unsafe(x, pardtype),
                self.target,
            )

        k = jax.random.PRNGKey(seed)
        k1, k2, k3, k4, k5 = jax.random.split(k, 5)

        self.samples = distribute_to_devices_along_axis(
            jax.random.normal(k1, (n_samp, 2))
        )
        self.w = distribute_to_devices_along_axis(
            jax.random.normal(k2, (n_samp,), self.dtype).astype(self.dtype)
        )  # TODO remove astype once its fixed in jax
        self.params = tree_random_normal_like(k3, self.target)
        self.v = tree_random_normal_like(k4, self.target)
        self.grad = tree_random_normal_like(k5, self.target)

        if holomorphic:

            @partial(jax.vmap, in_axes=(None, 0))
            def f(params, x):
                return astype_unsafe(
                    params["a"][0][0][0] * x[0]
                    + params["b"] * x[1]
                    + params["c"] * (x[0] * x[1])
                    + jnp.sin(x[1] * params["a"][0][1][0])
                    * jnp.cos(x[0] * params["b"] + 1j)
                    * params["c"],
                    self.dtype,
                )

        else:

            @partial(jax.vmap, in_axes=(None, 0))
            def f(params, x):
                return astype_unsafe(
                    params["a"][0][0][0].conjugate() * x[0]
                    + params["b"] * x[1]
                    + params["c"] * (x[0] * x[1])
                    + jnp.sin(x[1] * params["a"][0][1][0])
                    * jnp.cos(x[0] * params["b"].conjugate() + 1j)
                    * params["c"].conjugate(),
                    self.dtype,
                )

        self.f = Partial(f)

        self.params_real_flat = tree_toreal_flat(self.params)
        self.grad_real_flat = tree_toreal_flat(self.grad)
        self.v_real_flat = tree_toreal_flat(self.v)
        self.ok_real = self.grads_real(self.params_real_flat, self.samples)
        self.okmean_real = self.ok_real.mean(axis=0)
        self.dok_real = self.ok_real - self.okmean_real
        self.S_real = (
            self.dok_real.conjugate().transpose() @ self.dok_real / n_samp
        ).real
        self.scale = jnp.sqrt(self.S_real.diagonal() + offset)
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


@common.named_parametrize("holomorphic", [True])
@common.named_parametrize("n_samp", [0])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_reassemble_complex(e):
    assert_tree_allclose(
        e.params, reassemble_complex(tree_toreal_flat(e.params), target=e.target)
    )


@common.named_parametrize("holomorphic", [True, False])
@common.named_parametrize("n_samp", [24 * device_count_per_rank(), 1024])
@common.named_parametrize("jit", [True, False])
@pytest.mark.parametrize("outdtype, pardtype", all_test_types)
@common.named_parametrize("chunk_size", [8, None])
def test_matvec(e, jit, chunk_size):
    diag_shift = 0.01

    def f(params_model_state, x):
        return e.f(params_model_state["params"], x)

    samples = e.samples
    if chunk_size is None:
        mat_vec_factory = qgt_onthefly_logic.mat_vec_factory

    else:
        mat_vec_factory = partial(
            qgt_onthefly_logic.mat_vec_chunked_factory, chunk_size=chunk_size
        )

    mv = mat_vec_factory(f, e.params, {}, samples)
    if jit:
        mv = jax.jit(mv)
    actual = mv(e.v, diag_shift)
    expected = reassemble_complex(
        e.S_real @ e.v_real_flat + diag_shift * e.v_real_flat, target=e.target
    )
    assert_tree_allclose(actual, expected)
    tree_samedtypes(actual, expected)


@common.named_parametrize("holomorphic", [True, False])
@common.named_parametrize("n_samp", [24 * device_count_per_rank(), 1024])
@common.named_parametrize("jit", [True, False])
@pytest.mark.parametrize("outdtype, pardtype", all_test_types)
@common.named_parametrize("chunk_size", [8, None])
def test_matvec_linear_transpose(e, jit, chunk_size):
    def f(params_model_state, x):
        return e.f(params_model_state["params"], x)

    samples = e.samples

    if chunk_size is None:
        mat_vec_factory = qgt_onthefly_logic.mat_vec_factory
    else:
        mat_vec_factory = partial(
            qgt_onthefly_logic.mat_vec_chunked_factory, chunk_size=chunk_size
        )

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
    assert_tree_allclose(actual, expected)
    tree_samedtypes(actual, expected)


# TODO separate test for prepare_centered_oks
@common.named_parametrize("holomorphic", [True])
@common.named_parametrize("n_samp", [25 * device_count_per_rank(), 1024])
@common.named_parametrize("jit", [True, False])
@common.named_parametrize("chunk_size", [7, None])
@pytest.mark.parametrize(
    "outdtype, pardtype", r_r_test_types + c_c_test_types + r_c_test_types
)
def test_matvec_treemv(e, jit, holomorphic, pardtype, outdtype, chunk_size):
    mv = qgt_jacobian_pytree._mat_vec

    if not nkjax.is_complex_dtype(pardtype) and nkjax.is_complex_dtype(outdtype):
        jacobian_fun = nkjax.compose(
            nkjax._jacobian.jacobian_pytree.stack_jacobian_tuple,
            nkjax._jacobian.jacobian_pytree.jacobian_cplx,
        )
    else:
        jacobian_fun = nkjax._jacobian.jacobian_pytree.jacobian_real_holo

    jacobian_fun = nkjax.vmap_chunked(
        jacobian_fun, in_axes=(None, None, 0), chunk_size=chunk_size
    )
    centered_jacobian_fun = nkjax.compose(tree_subtract_mean, jacobian_fun)

    if jit:
        mv = jax.jit(mv)
        centered_jacobian_fun = jax.jit(centered_jacobian_fun)

    centered_oks = centered_jacobian_fun(e.f, e.params, e.samples)
    centered_oks = divide_by_sqrt_n_samp(centered_oks, e.samples)
    actual = mv(e.v, centered_oks)
    expected = reassemble_complex(e.S_real @ e.v_real_flat, target=e.target)
    assert_tree_allclose(actual, expected)
    tree_samedtypes(actual, expected)


# TODO separate test for prepare_centered_oks
# TODO test C->R ?
@common.named_parametrize("holomorphic", [True, False])
@common.named_parametrize("n_samp", [25 * device_count_per_rank(), 1024])
@common.named_parametrize("jit", [True, False])
@pytest.mark.parametrize("outdtype, pardtype", test_types)
def test_matvec_treemv_modes(e, jit, holomorphic, pardtype, outdtype):
    diag_shift = 0.01
    model_state = {}

    def apply_fun(params, samples):
        return e.f(params["params"], samples)

    mv = qgt_jacobian_pytree.mat_vec

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

    centered_oks = nkjax.jacobian(
        apply_fun,
        e.params,
        e.samples,
        model_state,
        mode=mode,
        dense=False,
        center=True,
        _sqrt_rescale=True,
    )
    # TODO Apply offset if offset is not None

    actual = reassemble(mv(v, centered_oks, diag_shift))
    expected = reassemble_complex(
        e.S_real @ e.v_real_flat + diag_shift * e.v_real_flat, target=e.target
    )
    assert_tree_allclose(actual, expected)
    tree_samedtypes(actual, expected)


@pytest.fixture
def e_offset(n_samp, outdtype, pardtype, holomorphic, offset, seed=123):
    return Example(n_samp, seed, outdtype, pardtype, holomorphic, offset)


@pytest.mark.parametrize("holomorphic", [True])
@pytest.mark.parametrize("n_samp", [25 * device_count_per_rank(), 1024])
@pytest.mark.parametrize(
    "outdtype, pardtype",
    r_c_test_types,  # r_r_test_types + c_c_test_types + r_c_test_types
)
@pytest.mark.parametrize("offset", [0.0, 0.1])
def test_scale_invariant_regularization(e_offset, outdtype, pardtype, offset):
    e = e_offset
    mv = qgt_jacobian_pytree._mat_vec

    if not nkjax.is_complex_dtype(pardtype) and nkjax.is_complex_dtype(outdtype):
        jacobian_fun = nkjax.compose(
            nkjax._jacobian.jacobian_pytree.stack_jacobian_tuple,
            nkjax._jacobian.jacobian_pytree.jacobian_cplx,
        )
        ndims = 2
    else:
        jacobian_fun = nkjax._jacobian.jacobian_pytree.jacobian_real_holo
        ndims = 1

    jacobian_fun = jax.vmap(jacobian_fun, in_axes=(None, None, 0))
    centered_jacobian_fun = nkjax.compose(tree_subtract_mean, jacobian_fun)
    centered_oks = centered_jacobian_fun(e.f, e.params, e.samples)
    centered_oks = divide_by_sqrt_n_samp(centered_oks, e.samples)

    centered_oks_scaled, scale = qgt_jacobian_common.rescale(
        centered_oks, offset, ndims=ndims
    )
    actual = mv(e.v, centered_oks_scaled)
    expected = reassemble_complex(e.S_real_scaled @ e.v_real_flat, target=e.target)

    rtol = 1e-5 if pardtype is jnp.float32 else 1e-8
    assert_tree_allclose(actual, expected, rtol=rtol)
    tree_samedtypes(actual, expected)


def test_qgt_onthefly_dense_nonholo_error():
    import netket as nk

    hi = nk.hilbert.Spin(0.5, 2)
    ma = nk.models.RBM(param_dtype=complex)
    vs = nk.vqs.FullSumState(hi, ma)
    with pytest.raises(nk.errors.NonHolomorphicQGTOnTheFlyDenseRepresentationError):
        qgt = nk.optimizer.qgt.QGTOnTheFly(vs)
        qgt.to_dense()

    with pytest.raises(nk.errors.NonHolomorphicQGTOnTheFlyDenseRepresentationError):
        qgt = nk.optimizer.qgt.QGTOnTheFly(vs, holomorphic=False)
        qgt.to_dense()


@common.named_parametrize("dense", [False, True])
def test_qgt_jacobian_imaginary(dense):
    g = nk.graph.Hypercube(length=6, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

    ma = nk.models.RBM(alpha=1, param_dtype=complex)

    sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
    vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

    E, F = vs.expect_and_grad(ha)

    if dense:
        qgt = nk.optimizer.qgt.QGTJacobianDense(diag_shift=0.00)
    else:
        qgt = nk.optimizer.qgt.QGTJacobianPyTree(diag_shift=0.00)

    Sr = qgt(vs, mode="complex")
    Si = qgt(vs, mode="imag")
    Sh = qgt(vs, mode="holomorphic")

    solver = nk.optimizer.solver.pinv

    xr, _ = Sr.solve(solver, F)
    xi, _ = Si.solve(solver, F)
    xh, _ = Sh.solve(solver, F)

    xrd, _ = nk.jax.tree_ravel(xr)
    xid, _ = nk.jax.tree_ravel(xi)
    xhd, _ = nk.jax.tree_ravel(xh)

    np.testing.assert_allclose(Sr @ xrd, Si @ xid, rtol=1e-5)

    shd = Sh.to_dense()
    srd = Sr.to_dense()
    sid = Si.to_dense()
    n = shd.shape[0]
    np.testing.assert_allclose(srd[:n, :n], shd.real)
    np.testing.assert_allclose(srd[n:, :n], shd.imag, atol=1e-16)
    np.testing.assert_allclose(srd[:n, n:], -shd.imag, atol=1e-16)
    np.testing.assert_allclose(srd[n:, n:], shd.real, atol=1e-16)

    np.testing.assert_allclose(sid[:n, :n], shd.imag, atol=1e-16)
    np.testing.assert_allclose(sid[:n, n:], shd.real)
    np.testing.assert_allclose(sid[n:, :n], -shd.real)
    np.testing.assert_allclose(sid[n:, n:], shd.imag, atol=1e-16)


def test_qgt_jacobian_imaginary_match():
    g = nk.graph.Hypercube(length=6, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

    ma = nk.models.RBM(alpha=1, param_dtype=complex)

    sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
    vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

    Sd = nk.optimizer.qgt.QGTJacobianDense(vs, diag_shift=0.00, mode="imag")
    Sp = nk.optimizer.qgt.QGTJacobianPyTree(vs, diag_shift=0.00, mode="imag")

    np.testing.assert_allclose(Sd.to_dense(), Sp.to_dense(), atol=1e-15)

    E, F = vs.expect_and_grad(ha)
    F, _ = nk.jax.tree_ravel(F)
    xd = Sd @ F
    xp = Sp @ F
    np.testing.assert_allclose(xd, xp, atol=1e-15)


def test_qgt_jacobian_imaginary_conversion():
    # Check the to_real/to_imag functions
    g = nk.graph.Hypercube(length=3, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

    ma = nk.models.RBM(alpha=1, param_dtype=complex)

    sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
    vs = nk.vqs.MCState(sa, ma, n_samples=512, n_discard_per_chain=0)

    for QGT_T in [
        nk.optimizer.qgt.QGTJacobianDense,
        nk.optimizer.qgt.QGTJacobianPyTree,
    ]:
        S_real = QGT_T(vs, diag_shift=1.00, mode="complex")
        S_imag_conv = S_real.to_imag_part()

        S_imag = QGT_T(vs, diag_shift=1.00, mode="imag")

        S1_leaves, S1_treedef = jax.tree.flatten(S_imag_conv)
        S2_leaves, S2_treedef = jax.tree.flatten(S_imag)

        assert S1_treedef == S2_treedef
        for S1l, S2l in zip(S1_leaves, S2_leaves):
            np.testing.assert_allclose(S1l, S2l)


# TODO test with MPI
