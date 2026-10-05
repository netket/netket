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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import netket as nk
from netket.experimental.operator import SpinExchangeOperator
from netket.vqs.mc import kernels


def _j1j2(L):
    g = nk.graph.Square(L, max_neighbor_order=2, pbc=True)
    J = np.where(np.asarray([c for *_, c in g.edges(return_color=True)]) == 0, 1, 0.5)
    return g, J


# Small systems: the operators are compared through their sparse matrices.
GRAPHS = {
    "chain": lambda: (nk.graph.Chain(8, pbc=True), 1.0),
    "square": lambda: (nk.graph.Grid([4, 2], pbc=[True, False]), 1.0),
    "triangular": lambda: (nk.graph.Triangular([3, 4], pbc=True), 1.0),
    "j1j2": lambda: _j1j2(4),
}


def _assert_sparse_close(a, b):
    assert abs(a.to_sparse() - b.to_sparse()).max() < 1e-12


def _reference(hi, edges, J, Jz, h, sign_rule):
    sz_sz = np.diag([1.0, -1, -1, 1])
    exchange = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0.0]])
    J = np.broadcast_to(J, len(edges))
    Jz = np.broadcast_to(Jz, len(edges))
    h = np.broadcast_to(h, hi.size)
    H = nk.operator.LocalOperator(hi)
    for (i, j), Je, Jze in zip(edges, J, Jz):
        Je = -Je if sign_rule else Je
        H += nk.operator.LocalOperator(hi, Je * exchange + Jze * sz_sz, [i, j])
    for i in range(hi.size):
        H += h[i] * nk.operator.spin.sigmaz(hi, i)
    return H


@pytest.mark.parametrize("graph", list(GRAPHS))
@pytest.mark.parametrize("total_sz", [None, 0, 2])
@pytest.mark.parametrize("couplings", ["heisenberg", "xxz"])
def test_spin_exchange_matrix(graph, total_sz, couplings):
    g, J = GRAPHS[graph]()
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=total_sz)
    edges = np.asarray(g.edges())
    if couplings == "heisenberg":
        kwargs = {"J": J}
        expected = _reference(hi, edges, J, J, 0.0, g.is_bipartite())
        if not isinstance(J, np.ndarray):
            _assert_sparse_close(expected, nk.operator.Heisenberg(hi, g))
    else:
        rng = np.random.default_rng(0)
        J = J * rng.uniform(0.5, 1.5, len(edges))
        Jz = rng.uniform(-1, 1, len(edges))
        h = rng.uniform(-1, 1, hi.size)
        kwargs = {"J": J, "Jz": Jz, "h": h, "sign_rule": False}
        expected = _reference(hi, edges, J, Jz, h, False)

    H = SpinExchangeOperator(hi, g, **kwargs)
    _assert_sparse_close(H, expected)
    _assert_sparse_close(H.to_local_operator(), expected)

    # The bound holds for every configuration, and the diagonal comes first.
    x = hi.all_states()
    xp, mels = jax.jit(lambda H, x: H.get_conn_padded(x))(H, x)
    assert xp.shape == (hi.n_states, H.max_offdiag_conn_size + 1, hi.size)
    assert not np.any(np.isnan(mels))
    np.testing.assert_array_equal(xp[:, 0], x)
    np.testing.assert_allclose(mels[:, 0], expected.to_sparse().diagonal(), atol=1e-13)
    assert np.all((mels[:, 1:] != 0).sum(-1) <= H.max_offdiag_conn_size)


@pytest.mark.parametrize(
    "graph, total_sz, bound",
    [
        # bipartite: the Néel state has all the bonds anti-parallel
        (nk.graph.Chain(10, pbc=True), 0, 10),
        (nk.graph.Square(4, pbc=True), 0, 32),
        # at most 2 of the 3 bonds of every triangle are anti-parallel
        (nk.graph.Triangular([3, 4], pbc=True), 0, 24),
        (nk.graph.Triangular([3, 4], pbc=True), None, 24),
        # 2 down spins have at most 2 * 4 anti-parallel bonds
        (nk.graph.Square(4, pbc=True), 6, 8),
    ],
)
def test_spin_exchange_bound(graph, total_sz, bound):
    hi = nk.hilbert.Spin(0.5, graph.n_nodes, total_sz=total_sz)
    H = SpinExchangeOperator(hi, graph)
    assert H.max_offdiag_conn_size == bound
    assert H.max_conn_size == bound + 1
    # the bounds are attained
    _, mels = H.get_conn_padded(hi.all_states())
    assert np.max((mels[:, 1:] != 0).sum(-1)) == bound


def test_spin_exchange_overflow():
    g = nk.graph.Chain(8, pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    H = SpinExchangeOperator(hi, g, max_offdiag_conn_size=4)
    x = hi.all_states()
    _, mels = H.get_conn_padded(x)
    _, mels_ref = SpinExchangeOperator(hi, g).get_conn_padded(x)
    n_offdiag = (mels_ref[:, 1:] != 0).sum(-1)
    np.testing.assert_array_equal(np.isnan(mels[:, 0]), n_offdiag > 4)


@pytest.mark.parametrize("graph", ["chain", "triangular", "j1j2"])
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_spin_exchange_local_estimators(graph, chunk_size):
    g, J = GRAPHS[graph]()
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    H = SpinExchangeOperator(hi, g, J=J)
    H_ref = H.to_local_operator()

    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=8 * jax.device_count())
    ma = nk.models.RBM(alpha=2, param_dtype=complex)
    vs = nk.vqs.MCState(sa, ma, n_samples=64 * jax.device_count(), seed=0)

    expected = np.asarray(vs.local_estimators(H_ref, chunk_size=chunk_size).data)
    result = np.asarray(vs.local_estimators(H, chunk_size=chunk_size).data)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vs.expect(H).mean, vs.expect(H_ref).mean, rtol=1e-12)


def _vstate(hi, g):
    sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=8 * jax.device_count())
    ma = nk.models.RBM(alpha=2, param_dtype=complex)
    return nk.vqs.MCState(sa, ma, n_samples=64 * jax.device_count(), seed=0)


def test_spin_exchange_coupling_gradient_at_zero():
    # At J = 0 the exchange matrix elements are zero, but their derivative with
    # respect to J is not: the bonds must not be dropped based on the value of J.
    g = nk.graph.Triangular([3, 4], pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    vs = _vstate(hi, g)
    σ = vs.samples.reshape(-1, hi.size)

    def loss(J):
        H = SpinExchangeOperator(hi, g, J=J, Jz=1.0)
        out = kernels.local_value_kernel_jax(vs._apply_fun, vs.variables, σ, H)
        return jnp.sum(out).real

    J = jnp.array(0.0)
    result = jax.jit(jax.grad(loss))(J)
    eps = 1e-5
    finite_diff = (loss(J + eps) - loss(J - eps)) / (2 * eps)
    assert abs(result) > 1
    np.testing.assert_allclose(result, finite_diff, rtol=1e-6)

    # the bonds of a coupling given as a jax array are kept also when zero
    H = SpinExchangeOperator(hi, g, J=jnp.zeros(g.n_edges), Jz=1.0)
    assert H.max_offdiag_conn_size == SpinExchangeOperator(hi, g).max_offdiag_conn_size
    # those given as a Python or numpy zero are dropped
    H = SpinExchangeOperator(hi, g, J=0.0, Jz=1.0)
    assert H.max_offdiag_conn_size == 0
    _assert_sparse_close(H, _reference(hi, np.asarray(g.edges()), 0.0, 1.0, 0.0, False))


@pytest.mark.parametrize(
    "case", ["no_bonds", "no_exchange", "polarized", "polarized_no_bonds"]
)
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_spin_exchange_diagonal_only(case, chunk_size):
    # Without exchange (or anti-parallel bonds) the operator is diagonal.
    g = nk.graph.Chain(8, pbc=True)
    edges = np.asarray(g.edges())
    total_sz = 4 if case.startswith("polarized") else 0
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=total_sz)
    h = np.linspace(-1, 1, hi.size)
    J = 1.0
    if case.endswith("no_bonds"):
        edges = np.zeros((0, 2), dtype=int)
    if case == "no_exchange":
        J = 0.0
    H = SpinExchangeOperator(hi, edges, J=J, Jz=0.5, h=h)
    assert H.max_offdiag_conn_size == 0
    assert H.max_conn_size == 1
    expected = _reference(hi, edges, J, 0.5, h, False)
    _assert_sparse_close(H, expected)

    x = hi.all_states()
    xp, mels = jax.jit(lambda H, x: H.get_conn_padded(x))(H, x)
    np.testing.assert_array_equal(xp[:, 0], x)
    np.testing.assert_allclose(mels[:, 0], expected.to_sparse().diagonal(), atol=1e-13)

    if hi.n_states > 1:
        vs = _vstate(hi, nk.graph.Chain(8, pbc=True))
        np.testing.assert_allclose(
            np.asarray(vs.local_estimators(H, chunk_size=chunk_size).data),
            np.asarray(vs.local_estimators(expected, chunk_size=chunk_size).data),
            rtol=1e-12,
            atol=1e-12,
        )


def test_spin_exchange_local_estimators_bitwise_repeatable():
    # Repeated evaluations on the same inputs must give the same bits.
    g = nk.graph.Triangular([3, 4], pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    H = SpinExchangeOperator(hi, g)
    vs = _vstate(hi, g)
    σ = vs.samples.reshape(-1, hi.size)
    f = jax.jit(lambda v, σ, H: kernels.local_value_kernel_jax(vs._apply_fun, v, σ, H))
    expected = np.asarray(f(vs.variables, σ, H)).tobytes()
    for _ in range(20):
        assert np.asarray(f(vs.variables, σ, H)).tobytes() == expected
