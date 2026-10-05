# Copyright 2026 The NetKet Authors - All rights reserved.
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

"""Tests for CompactConnOperator and the compact local-value kernel."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import netket as nk
from netket.experimental.operator import CompactConnOperator
from netket.vqs.mc import kernels
from netket._src.operator.compact_conn import split_diagonal


def _heisenberg_chain():
    g = nk.graph.Chain(12, pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    H = nk.operator.Heisenberg(hi, g)
    # Every bond can be anti-parallel (Néel state).
    return hi, H, g.n_edges, nk.sampler.MetropolisExchange(hi, graph=g)


def _heisenberg_triangular():
    g = nk.graph.Triangular([3, 4], pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    H = nk.operator.Heisenberg(hi, g)
    # At most 2 of the 3 bonds of every triangle are anti-parallel.
    return hi, H, 2 * g.n_nodes, nk.sampler.MetropolisExchange(hi, graph=g)


def _ising():
    g = nk.graph.Chain(10, pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes)
    H = nk.operator.IsingJax(hi, g, h=1.0)
    return hi, H, g.n_nodes, nk.sampler.MetropolisLocal(hi)


def _hubbard():
    g = nk.graph.Square(3, pbc=True)
    hi = nk.hilbert.SpinOrbitalFermions(g.n_nodes, s=1 / 2, n_fermions_per_spin=(2, 2))
    c, cdag = nk.operator.fermion.destroy, nk.operator.fermion.create
    nc = nk.operator.fermion.number
    H = 0.0
    for sz in (-1, 1):
        for i, j in g.edges():
            H -= cdag(hi, i, sz) @ c(hi, j, sz) + cdag(hi, j, sz) @ c(hi, i, sz)
    for i in g.nodes():
        H += 4.0 * nc(hi, i, -1) @ nc(hi, i, 1)
    # Per spin, at most one of the two hoppings of every bond is nonzero.
    return (
        hi,
        H.to_jax_operator(),
        2 * g.n_edges,
        nk.sampler.MetropolisFermionHop(hi, graph=g),
    )


SYSTEMS = {
    "heisenberg_chain": _heisenberg_chain,
    "heisenberg_triangular": _heisenberg_triangular,
    "ising": _ising,
    "hubbard": _hubbard,
}


def _vstate(sa):
    sa = sa.replace(n_chains=8 * jax.device_count())
    ma = nk.models.RBM(alpha=2, param_dtype=complex)
    return nk.vqs.MCState(sa, ma, n_samples=64 * jax.device_count(), seed=0)


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_compact_conn_operator(system):
    hi, H, K, _ = SYSTEMS[system]()
    Hc = CompactConnOperator(H, K)
    assert Hc.max_offdiag_conn_size == K
    assert Hc.max_conn_size == K + 1
    assert Hc.hilbert == H.hilbert
    assert Hc.is_hermitian == H.is_hermitian
    assert Hc.validate() <= K
    np.testing.assert_allclose(Hc.to_dense(), H.to_dense(), atol=1e-14)

    # the diagonal comes first
    x = hi.all_states()[:20]
    xp, mels = Hc.get_conn_padded(x)
    np.testing.assert_array_equal(xp[:, 0], x)
    np.testing.assert_allclose(mels[:, 0], np.diag(H.to_dense())[:20], atol=1e-14)

    # it is a pytree
    xp2, mels2 = jax.jit(lambda H, x: H.get_conn_padded(x))(Hc, x)
    np.testing.assert_array_equal(xp2, xp)
    np.testing.assert_array_equal(mels2, mels)


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("chunk_size", [None, 7, 64])
def test_compact_kernel_matches_padded(system, chunk_size):
    _, H, K, sa = SYSTEMS[system]()
    vs = _vstate(sa)
    Hc = CompactConnOperator(H, K)
    assert nk.vqs.get_local_kernel(vs, Hc) is kernels.local_value_kernel_jax_compact
    expected = np.asarray(vs.local_estimators(H).data)
    result = np.asarray(vs.local_estimators(Hc, chunk_size=chunk_size).data)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_compact_kernel_gradient():
    _, H, K, sa = SYSTEMS["hubbard"]()
    vs = _vstate(sa)
    Hc = CompactConnOperator(H, K)
    σ = vs.samples.reshape(-1, vs.hilbert.size)

    def loss(kernel, params, H):
        variables = {**vs.variables, "params": params}
        return jnp.sum(jnp.abs(kernel(vs._apply_fun, variables, σ, H)) ** 2)

    expected = jax.jit(
        jax.grad(lambda p, H: loss(kernels.local_value_kernel_jax, p, H))
    )(vs.parameters, H)
    result = jax.jit(
        jax.grad(lambda p, H: loss(kernels.local_value_kernel_jax_compact, p, H))
    )(vs.parameters, Hc)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12),
        result,
        expected,
    )


def test_compact_conn_overflow():
    hi, H, K, sa = SYSTEMS["heisenberg_triangular"]()
    Hc = CompactConnOperator(H, 4)
    with pytest.raises(ValueError, match="more than max_offdiag_conn_size=4"):
        Hc.validate()

    vs = _vstate(sa)
    with pytest.warns(RuntimeWarning, match="its local estimator is NaN"):
        e = np.asarray(vs.local_estimators(Hc).data)
        jax.effects_barrier()
    σ = vs.samples.reshape(-1, hi.size)
    n_offdiag = np.asarray(jax.jit(lambda H, σ: split_diagonal(H, σ)[3].sum(-1))(H, σ))
    np.testing.assert_array_equal(np.isnan(e.reshape(-1)), n_offdiag > 4)

    _, mels = Hc.get_conn_padded(σ)
    np.testing.assert_array_equal(np.isnan(mels[:, 0]), n_offdiag > 4)


def test_operator_declared_bound():
    # An operator class can declare an analytic bound, which is then used by
    # default, both by the wrapper and by the local estimators.
    hi, H, K, sa = SYSTEMS["heisenberg_triangular"]()

    @jax.tree_util.register_pytree_node_class
    class BoundedOperator(type(H)):
        @property
        def max_offdiag_conn_size(self):
            return K

    Hb = H.copy()
    Hb.__class__ = BoundedOperator
    assert CompactConnOperator(Hb).max_offdiag_conn_size == K
    with pytest.raises(ValueError, match="does not declare"):
        CompactConnOperator(H)

    vs = _vstate(sa)
    assert nk.vqs.get_local_kernel(vs, H) is kernels.local_value_kernel_jax
    assert nk.vqs.get_local_kernel(vs, Hb) is kernels.local_value_kernel_jax_compact
    np.testing.assert_allclose(
        np.asarray(vs.local_estimators(Hb).data),
        np.asarray(vs.local_estimators(H).data),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_compact_kernel_coefficient_gradient_at_zero(chunk_size):
    # At h = 0 every off-diagonal matrix element of the Ising model is zero,
    # but its derivative with respect to h is not: the off-diagonal elements
    # must be selected from the configurations, not from their matrix elements.
    _, _, K, sa = SYSTEMS["ising"]()
    vs = _vstate(sa)
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    graph = nk.graph.Chain(vs.hilbert.size, pbc=True)

    def compact(*args):
        return kernels.local_value_kernel_jax_compact(*args, chunk_size=chunk_size)

    def loss(kernel, h, wrap):
        H = nk.operator.IsingJax(vs.hilbert, graph, h=h, J=1.0)
        if wrap:
            H = CompactConnOperator(H, K)
        return jnp.sum(kernel(vs._apply_fun, vs.variables, σ, H)).real

    h = jnp.array(0.0)
    padded = kernels.local_value_kernel_jax
    expected = jax.jit(jax.grad(lambda h: loss(padded, h, False)))(h)
    eps = 1e-5
    finite_diff = (loss(compact, h + eps, True) - loss(compact, h - eps, True)) / (
        2 * eps
    )
    assert abs(expected) > 1
    np.testing.assert_allclose(finite_diff, expected, rtol=1e-6)
    # the compact kernel, and the padded kernel on the compact operator
    for kernel in (compact, padded):
        result = jax.jit(jax.grad(lambda h: loss(kernel, h, True)))(h)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


def _empty_operator(name):
    if name == "local_operator":
        hi = nk.hilbert.Spin(0.5, 4)
        H = nk.operator.LocalOperatorJax(hi)
        sa = nk.sampler.MetropolisLocal(hi)
    else:
        hi = nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)
        H = nk.operator.FermionOperator2ndJax(hi)
        sa = nk.sampler.MetropolisFermionHop(hi, graph=nk.graph.Chain(4))
    return H, sa


@pytest.mark.parametrize("operator", ["local_operator", "fermion_operator"])
@pytest.mark.parametrize("K", [0, 2])
@pytest.mark.parametrize("chunk_size", [None, 4])
def test_compact_kernel_empty_operator(operator, K, chunk_size):
    # An operator without connected elements is a valid zero operator.
    H, sa = _empty_operator(operator)
    assert H.max_conn_size == 0
    vs = _vstate(sa)
    Hc = CompactConnOperator(H, K)
    expected = np.asarray(vs.local_estimators(H).data)
    result = np.asarray(vs.local_estimators(Hc, chunk_size=chunk_size).data)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, 0)
    np.testing.assert_array_equal(expected, 0)

    σ = vs.samples.reshape(-1, vs.hilbert.size)
    xp, mels = Hc.get_conn_padded(σ)
    assert xp.shape == (σ.shape[0], K + 1, σ.shape[1])
    np.testing.assert_array_equal(mels, 0)

    def loss(params, H):
        variables = {**vs.variables, "params": params}
        out = kernels.local_value_kernel_jax_compact(
            vs._apply_fun, variables, σ, H, chunk_size=chunk_size
        )
        return jnp.sum(jnp.abs(out) ** 2)

    grad = jax.jit(jax.grad(loss))(vs.parameters, Hc)
    jax.tree.map(lambda g: np.testing.assert_array_equal(g, 0), grad)


@pytest.mark.parametrize("chunk_size", [None, 32])
def test_compact_kernel_bitwise_repeatable(chunk_size):
    # Repeated evaluations on the same inputs must give the same bits.
    _, H, K, sa = SYSTEMS["hubbard"]()
    vs = _vstate(sa)
    Hc = CompactConnOperator(H, K)
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    f = jax.jit(
        lambda v, σ, H: kernels.local_value_kernel_jax_compact(
            vs._apply_fun, v, σ, H, chunk_size=chunk_size
        )
    )
    expected = np.asarray(f(vs.variables, σ, Hc)).tobytes()
    for _ in range(20):
        assert np.asarray(f(vs.variables, σ, Hc)).tobytes() == expected
