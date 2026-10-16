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

"""Tests for the packed local-value kernel of jax operators."""

import os
import subprocess
import sys
import textwrap
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax.extend
import flax
from jax.flatten_util import ravel_pytree

import netket as nk
from netket.vqs.mc import kernels, kernels_packed

from .. import common


def _heisenberg_chain():
    g = nk.graph.Chain(12, pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    return hi, nk.operator.Heisenberg(hi, g), nk.sampler.MetropolisExchange(hi, graph=g)


def _heisenberg_triangular():
    g = nk.graph.Triangular([3, 4], pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    return hi, nk.operator.Heisenberg(hi, g), nk.sampler.MetropolisExchange(hi, graph=g)


def _j1j2_triangular():
    g = nk.graph.Triangular([3, 4], pbc=True, max_neighbor_order=2)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    H = nk.operator.Heisenberg(hi, g, J=[1.0, 0.4])
    return hi, H, nk.sampler.MetropolisExchange(hi, graph=g)


def _ising():
    g = nk.graph.Chain(10, pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes)
    return hi, nk.operator.IsingJax(hi, g, h=1.0), nk.sampler.MetropolisLocal(hi)


def _pauli_strings():
    hi = nk.hilbert.Spin(0.5, 8)
    strings = ["XXIIIIII", "YYIIIIII", "ZZIIIIII", "IXXIIIII", "IIZIIIXI", "IIIIYIIY"]
    weights = [1.0, 0.5, 0.2, -0.7, 0.3, 0.1]
    H = nk.operator.PauliStringsJax(hi, strings, weights)
    return hi, H, nk.sampler.MetropolisLocal(hi)


def _local_operator():
    # Terms acting on 1, 2 and 3 sites, with several connected elements per row.
    hi = nk.hilbert.Spin(0.5, 8)
    sx, sy, sz = (
        nk.operator.spin.sigmax,
        nk.operator.spin.sigmay,
        nk.operator.spin.sigmaz,
    )
    H = (
        sum(sx(hi, i) * sz(hi, (i + 1) % 8) for i in range(8))
        + 0.5
        * sum(sy(hi, i) * sy(hi, (i + 2) % 8) * sx(hi, (i + 5) % 8) for i in range(8))
        + 0.3 * sum(sx(hi, i) for i in range(8))
    )
    return hi, H.to_jax_operator(), nk.sampler.MetropolisLocal(hi)


def _hubbard(pnc):
    g = nk.graph.Square(2, pbc=True)
    hi = nk.hilbert.SpinOrbitalFermions(g.n_nodes, s=1 / 2, n_fermions_per_spin=(2, 2))
    if pnc:
        H = nk.operator.FermiHubbardJax(hi, g, t=1.0, U=4.0)
    else:
        c, cdag = nk.operator.fermion.destroy, nk.operator.fermion.create
        nc = nk.operator.fermion.number
        H = 0.0
        for sz in (-1, 1):
            for i, j in g.edges():
                H -= cdag(hi, i, sz) @ c(hi, j, sz) + cdag(hi, j, sz) @ c(hi, i, sz)
        for i in g.nodes():
            H += 4.0 * nc(hi, i, -1) @ nc(hi, i, 1)
        H = H.to_jax_operator()
    return hi, H, nk.sampler.MetropolisFermionHop(hi, graph=g)


SYSTEMS = {
    "heisenberg_chain": _heisenberg_chain,
    "heisenberg_triangular": _heisenberg_triangular,
    "j1j2_triangular": _j1j2_triangular,
    "ising": _ising,
    "pauli_strings": _pauli_strings,
    "local_operator": _local_operator,
    "hubbard": lambda: _hubbard(False),
    "hubbard_pnc": lambda: _hubbard(True),
}


def _vstate(system, param_dtype=complex, n_samples=64):
    hi, H, sa = SYSTEMS[system]()
    sa = sa.replace(n_chains=8 * jax.device_count())
    ma = nk.models.RBM(alpha=2, param_dtype=param_dtype)
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples * jax.device_count(), seed=0)
    # Set the operator up outside of any trace.
    jax.block_until_ready(H.get_conn_padded(vs.samples.reshape(-1, hi.size)[:1]))
    return vs, H


def _local_values(vs, H, chunk_size, packed):
    # The local values of the samples of vs, with the packed kernel or with the
    # padded one, called explicitly.
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    if packed:
        kernel = _packed(chunk_size)
    elif chunk_size is None:
        kernel = kernels.local_value_kernel_jax
    else:
        kernel = partial(kernels.local_value_kernel_jax_chunked, chunk_size=chunk_size)
    f = jax.jit(lambda v, σ, H: kernel(vs._apply_fun, v, σ, H))
    return np.asarray(f(vs.variables, σ, H))


def _packed(chunk_size=None, **kwargs):
    def kernel(*args):
        return kernels_packed.local_value_kernel_jax_packed(
            *args, chunk_size=chunk_size, **kwargs
        )

    return kernel


def _assert_close(a, b, rtol):
    # Relative to the largest element: the derivatives of a sum of many terms
    # are rounded relative to their scale, not to the element.
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) <= rtol * np.max(np.abs(b)), np.max(np.abs(a - b))


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("chunk_size", [None, 7, 64, 100000])
def test_packed_kernel_matches_padded(system, chunk_size):
    vs, H = _vstate(system)
    expected = _local_values(vs, H, None, packed=False)
    result = _local_values(vs, H, chunk_size, packed=True)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 100000])
@pytest.mark.parametrize("min_chunk_size", [None, 1, 3, 64])
def test_packed_kernel_min_chunk_size(chunk_size, min_chunk_size):
    vs, H = _vstate("heisenberg_triangular")
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    expected = kernels.local_value_kernel_jax(vs._apply_fun, vs.variables, σ, H)
    kernel = _packed(chunk_size, min_chunk_size=min_chunk_size)
    result = jax.jit(lambda v, σ, H: kernel(vs._apply_fun, v, σ, H))(vs.variables, σ, H)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("system", ["heisenberg_triangular", "ising", "hubbard"])
@pytest.mark.parametrize("conn_chunk_size", [None, 1, 50, 37 * 64])
def test_packed_kernel_conn_chunk_size(system, conn_chunk_size):
    # The samples are processed in groups (the last one padded), whose padded
    # connected configurations hold at most conn_chunk_size configurations.
    vs, H = _vstate(system)
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    expected = kernels.local_value_kernel_jax(vs._apply_fun, vs.variables, σ, H)
    kernel = _packed(16, conn_chunk_size=conn_chunk_size)
    result = jax.jit(lambda v, σ, H: kernel(vs._apply_fun, v, σ, H))(vs.variables, σ, H)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(jax.process_count() > 1, reason="counts the rows of every device")
@pytest.mark.parametrize(
    "chunk_size, min_chunk_size, multiple",
    [(32, None, 4), (32, 1, 1), (32, 32, 32), (48, None, 4), (48, 48, 48)],
)
@pytest.mark.parametrize("system", ["heisenberg_triangular", "ising"])
def test_packed_kernel_evaluated_rows(system, chunk_size, min_chunk_size, multiple):
    # On every device, the network is evaluated on the samples and on the
    # selected configurations, the last chunk being padded to min_chunk_size.
    # With the Ising model, almost all the chunks are full.
    vs, H = _vstate(system)
    rows = []

    def logpsi(variables, x):
        jax.debug.callback(lambda x: rows.append(x.shape[0]), x[:, 0])
        return vs._apply_fun(variables, x)

    kernel = _packed(chunk_size, min_chunk_size=min_chunk_size)
    f = jax.jit(lambda v, σ, H: kernel(logpsi, v, σ, H))
    remainders = set()
    for seed in range(4):
        σ = vs.hilbert.random_state(jax.random.key(seed), 61 * jax.device_count())
        σp, _ = H.get_conn_padded(σ)
        is_conn = np.any(np.asarray(σp) != np.asarray(σ)[:, None], axis=-1)
        expected = 0
        for n_conn in is_conn.reshape(jax.device_count(), -1).sum(axis=1):
            remainders.add(int(n_conn) % chunk_size)
            expected += 61 + -(-int(n_conn) // multiple) * multiple
        rows.clear()
        jax.block_until_ready(f(vs.variables, σ, H))
        jax.effects_barrier()
        assert sum(rows) == expected
    if system == "heisenberg_triangular":
        assert len(remainders) > 1


def _counting_ising(hilbert, graph, calls):
    # The Ising model, recording the number of rows of every call to
    # get_conn_padded.
    @jax.tree_util.register_pytree_node_class
    class CountingIsing(nk.operator.IsingJax):
        def get_conn_padded(self, x):
            jax.debug.callback(lambda r: calls["padded"].append(len(r)), x[:, 0])
            return super().get_conn_padded(x)

    return CountingIsing(hilbert, graph, h=1.0)


@pytest.mark.skipif(jax.process_count() > 1, reason="counts the rows of every device")
@pytest.mark.parametrize("conn_chunk_size", [2, 12, 40])
@pytest.mark.parametrize("chunk_size", [None, 7, 64])
def test_packed_kernel_conn_chunk_size_bound(conn_chunk_size, chunk_size):
    # The connected configurations are computed at most conn_chunk_size at a
    # time, in groups of samples, or those of a single sample (max_conn_size =
    # 5) when it has more.
    calls = {"padded": []}
    hi = nk.hilbert.Spin(0.5, 4)
    graph = nk.graph.Chain(4)
    H = _counting_ising(hi, graph, calls)
    σ = hi.random_state(jax.random.key(0), 32 * jax.device_count())
    p = jnp.array([0.1, 0.2, -0.3, 0.4])

    def logpsi(p, x):
        return x @ p

    kernel = _packed(chunk_size, conn_chunk_size=conn_chunk_size)
    result = jax.jit(lambda p, σ, H: kernel(logpsi, p, σ, H))(p, σ, H)
    jax.block_until_ready(result)
    jax.effects_barrier()
    expected = kernels.local_value_kernel_jax(
        lambda p, x: x @ p, p, σ, nk.operator.IsingJax(hi, graph, h=1.0)
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    assert calls["padded"]
    budget = max(conn_chunk_size, H.max_conn_size)
    assert all(n * H.max_conn_size <= budget for n in calls["padded"])


@pytest.mark.parametrize("chunk_size", [None, 32])
def test_packed_kernel_no_recompilation(chunk_size):
    # The number of nonzero connected elements changes from batch to batch,
    # but the kernel must be compiled only once.
    vs, H = _vstate("heisenberg_triangular")
    n_traces = 0

    def logpsi(variables, x):
        nonlocal n_traces
        n_traces += 1
        return vs._apply_fun(variables, x)

    kernel = _packed(chunk_size)

    @jax.jit
    def eloc(variables, σ, H):
        return kernel(logpsi, variables, σ, H)

    @jax.jit
    def n_nonzero(σ, H):
        return jnp.sum(H.n_conn(σ))

    counts = set()
    for _ in range(4):
        σ = vs.sample().reshape(-1, vs.hilbert.size)
        counts.add(int(n_nonzero(σ, H)))
        eloc(vs.variables, σ, H).block_until_ready()
        if _ == 0:
            n_traces_first = n_traces
    assert len(counts) > 1
    assert n_traces == n_traces_first


@pytest.mark.parametrize("system", ["hubbard", "heisenberg_triangular"])
@pytest.mark.parametrize("chunk_size", [None, 32])
def test_packed_kernel_gradient(system, chunk_size):
    # The chunks are selected by lax.cond and evaluated in loops with a static
    # number of iterations, which reverse-mode differentiation must go through.
    vs, H = _vstate(system)
    σ = vs.samples.reshape(-1, vs.hilbert.size)

    # The operator is passed as an argument: FermionOperator2ndJax must not be
    # set up inside of a trace.
    def loss(kernel, params, H):
        variables = {**vs.variables, "params": params}
        return jnp.sum(jnp.abs(kernel(vs._apply_fun, variables, σ, H)) ** 2)

    expected = jax.jit(
        jax.grad(lambda p, H: loss(kernels.local_value_kernel_jax, p, H))
    )(vs.parameters, H)
    result = jax.jit(jax.grad(lambda p, H: loss(_packed(chunk_size), p, H)))(
        vs.parameters, H
    )
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12),
        result,
        expected,
    )


@pytest.mark.parametrize("system", ["hubbard", "heisenberg_triangular"])
@pytest.mark.parametrize("chunk_size", [None, 32])
def test_packed_kernel_jvp(system, chunk_size):
    vs, H = _vstate(system)
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    tangent = jax.tree.map(lambda p: jnp.ones_like(p) * 0.1, vs.parameters)

    def f(kernel, params, H):
        return kernel(vs._apply_fun, {**vs.variables, "params": params}, σ, H)

    expected = jax.jit(
        lambda p, t, H: jax.jvp(
            lambda p: f(kernels.local_value_kernel_jax, p, H), (p,), (t,)
        )
    )(vs.parameters, tangent, H)
    result = jax.jit(
        lambda p, t, H: jax.jvp(lambda p: f(_packed(chunk_size), p, H), (p,), (t,))
    )(vs.parameters, tangent, H)
    for a, b in zip(result, expected):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


def test_packed_kernel_expect_gradient():
    # The packed kernel as the integrand of nk.jax.expect, whose gradient also
    # differentiates the local values (like MCState.expect_and_grad with
    # use_covariance=False).
    vs, H = _vstate("heisenberg_chain")
    σ = vs.samples.reshape(-1, vs.hilbert.size)

    def expect(kernel, params, H):
        def log_pdf(params, σ):
            return 2 * vs._apply_fun({**vs.variables, "params": params}, σ).real

        def local_values(params, σ):
            return kernel(vs._apply_fun, {**vs.variables, "params": params}, σ, H)

        return nk.jax.expect(log_pdf, local_values, params, σ)[0].real

    f = jax.jit(jax.value_and_grad(expect, argnums=1), static_argnums=0)
    E0, g0 = f(kernels.local_value_kernel_jax, vs.parameters, H)
    E1, g1 = f(_packed(16), vs.parameters, H)
    np.testing.assert_allclose(E1, E0, rtol=1e-12)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12), g1, g0
    )


@pytest.mark.parametrize("conn_chunk_size", [None, 37 * 16])
def test_packed_kernel_logpsi_of_samples_after_loops(conn_chunk_size):
    # log ψ(σ) of the samples is computed with the same function and inputs as
    # the forward pass of the vjp of expect_and_grad, so that XLA merges them.
    # If a loop of the kernel takes it as an input, the intermediates of that
    # forward pass, which the vjp keeps for the backward pass, stay in memory
    # during the whole kernel. log ψ(σ) must therefore only be used after the
    # loops over the groups of samples and over the chunks.
    vs, H = _vstate("heisenberg_triangular")
    assert H.max_conn_size == 37
    # 60 samples, a shape on which no chunk of connected configurations is
    # evaluated: 4 groups of 15 samples with conn_chunk_size=37 * 16.
    σ = vs.samples.reshape(-1, vs.hilbert.size)[:60]

    def logpsi(variables, x):
        return vs._apply_fun(variables, x)

    def kernel(variables, σ, H):
        return kernels_packed._local_value_kernel_jax_packed(
            logpsi,
            variables,
            σ,
            H,
            chunk_size=None,
            min_chunk_size=None,
            conn_chunk_size=conn_chunk_size,
        )

    jaxpr = jax.make_jaxpr(kernel)(vs.variables, σ, H).jaxpr
    on_samples = [
        eqn
        for eqn in jaxpr.eqns
        if eqn.params.get("name") == "logpsi"
        and any(getattr(v.aval, "shape", None) == σ.shape for v in eqn.invars)
    ]
    assert len(on_samples) == 1
    # The values computed from log ψ(σ), none of which may enter an equation
    # that contains a loop or a cond (directly, or as apply_packed does).
    depends = set(on_samples[0].outvars)
    n_loops = 0
    for eqn in jaxpr.eqns:
        inputs = {v for v in eqn.invars if isinstance(v, jax.extend.core.Var)}
        loops = sum(
            _count(SimpleNamespace(eqns=[eqn]), name)
            for name in ("scan", "while", "cond")
        )
        n_loops += loops
        assert not (loops and inputs & depends), eqn.primitive.name
        if inputs & depends:
            depends.update(eqn.outvars)
    assert n_loops > 0

    expected = kernels.local_value_kernel_jax(vs._apply_fun, vs.variables, σ, H)
    result = jax.jit(kernel)(vs.variables, σ, H)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("system", ["heisenberg_triangular", "hubbard"])
@pytest.mark.parametrize("chunk_size", [None, 32])
def test_packed_kernel_config_flag(system, chunk_size, monkeypatch):
    # With NETKET_EXPERIMENTAL_PACKED_KERNEL, MCState.expect and expect_and_grad
    # compute the local values of jax operators with the packed kernel, with
    # the same results.
    vs, H = _vstate(system)
    vs.chunk_size = chunk_size
    assert not nk.config.netket_experimental_packed_kernel
    E0 = vs.expect(H)
    E0_, g0 = vs.expect_and_grad(H)

    packed = kernels_packed.local_value_kernel_jax_packed
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("chunk_size"))
        return packed(*args, **kwargs)

    monkeypatch.setattr(kernels_packed, "local_value_kernel_jax_packed", spy)
    with common.set_config("NETKET_EXPERIMENTAL_PACKED_KERNEL", True):
        E1 = vs.expect(H)
        assert calls == [chunk_size]
        E1_, g1 = vs.expect_and_grad(H)
        assert calls == [chunk_size, chunk_size]

    for a, b in [(E1, E0), (E1_, E0_)]:
        np.testing.assert_allclose(a.mean, b.mean, rtol=1e-12)
        np.testing.assert_allclose(a.variance, b.variance, rtol=1e-10)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12), g1, g0
    )


@common.onlyif_sharding_single_process
@pytest.mark.parametrize("system", ["heisenberg_triangular", "hubbard"])
def test_packed_kernel_no_collectives(system):
    # The packing must run per device: under GSPMD it would gather the
    # connected configurations of all devices.
    vs, H = _vstate(system)
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    kernel = _packed(None, conn_chunk_size=50)

    f = jax.jit(lambda v, σ, H: kernel(vs._apply_fun, v, σ, H))
    hlo = f.lower(vs.variables, σ, H).compile().as_text()
    for collective in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
        assert collective not in hlo


@pytest.mark.parametrize("system", list(SYSTEMS))
def test_padding_is_the_sample(system):
    # The kernel skips the connected configurations equal to the sample. Its
    # speedup relies on the operators filling the zero matrix elements with
    # the sample itself.
    vs, H = _vstate(system)
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    σp, mels = H.get_conn_padded(σ)
    is_conn = np.any(np.asarray(σp) != np.asarray(σ)[:, None, :], axis=-1)
    assert np.all(np.asarray(mels)[is_conn] != 0)


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_packed_kernel_coefficient_gradient_at_zero(chunk_size):
    # At h = 0 every off-diagonal matrix element of the Ising model is zero,
    # but its derivative with respect to h is not: the kernel must not drop
    # the connected configurations based on the value of their matrix element.
    vs, _ = _vstate("ising")
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    graph = nk.graph.Chain(vs.hilbert.size, pbc=True)
    packed = _packed(chunk_size)

    def loss(kernel, h):
        H = nk.operator.IsingJax(vs.hilbert, graph, h=h, J=1.0)
        return jnp.sum(kernel(vs._apply_fun, vs.variables, σ, H)).real

    h = jnp.array(0.0)
    expected = jax.jit(jax.grad(lambda h: loss(kernels.local_value_kernel_jax, h)))(h)
    result = jax.jit(jax.grad(lambda h: loss(packed, h)))(h)
    eps = 1e-5
    finite_diff = (loss(packed, h + eps) - loss(packed, h - eps)) / (2 * eps)

    assert abs(expected) > 1
    np.testing.assert_allclose(result, expected, rtol=1e-10)
    np.testing.assert_allclose(result, finite_diff, rtol=1e-6)


def _empty_operator(name):
    if name == "local_operator":
        hi = nk.hilbert.Spin(0.5, 4)
        H = nk.operator.LocalOperatorJax(hi)
        sa = nk.sampler.MetropolisLocal(hi)
    else:
        hi = nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)
        H = nk.operator.FermionOperator2ndJax(hi)
        sa = nk.sampler.MetropolisFermionHop(hi, graph=nk.graph.Chain(4))
    sa = sa.replace(n_chains=8 * jax.device_count())
    ma = nk.models.RBM(alpha=1, param_dtype=complex)
    vs = nk.vqs.MCState(sa, ma, n_samples=8 * jax.device_count(), seed=1)
    return vs, H


@pytest.mark.parametrize("operator", ["local_operator", "fermion_operator"])
@pytest.mark.parametrize("chunk_size", [None, 4])
def test_packed_kernel_empty_operator(operator, chunk_size):
    # An operator without connected elements is a valid zero operator.
    vs, H = _empty_operator(operator)
    assert H.max_conn_size == 0
    expected = _local_values(vs, H, chunk_size, packed=False)
    result = _local_values(vs, H, chunk_size, packed=True)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, 0)
    np.testing.assert_array_equal(expected, 0)

    σ = vs.samples.reshape(-1, vs.hilbert.size)
    kernel = _packed(chunk_size)

    def loss(params, H):
        variables = {**vs.variables, "params": params}
        return jnp.sum(jnp.abs(kernel(vs._apply_fun, variables, σ, H)) ** 2)

    grad = jax.jit(jax.grad(loss))(vs.parameters, H)
    jax.tree.map(lambda g: np.testing.assert_array_equal(g, 0), grad)


def test_packed_kernel_zero_samples_reproducer():
    # The reproducer of the review of https://github.com/netket/netket/pull/2293:
    # no samples, with an operator that has connected elements (unlike
    # test_packed_kernel_empty_operator).
    hi = nk.hilbert.Spin(0.5, N=4)
    H = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=1.0)
    p = jnp.array([0.1, 0.2, -0.3, 0.4])
    x = jnp.empty((0, 4))
    logpsi = lambda p, x: x @ p

    results = [
        jax.jit(lambda p, x: kernel(logpsi, p, x, H))(p, x)
        for kernel in (
            kernels.local_value_kernel_jax,
            kernels_packed.local_value_kernel_jax_packed,
        )
    ]
    for result in results:
        assert result.shape == (0,)
        assert result.dtype == results[0].dtype


@pytest.mark.parametrize("operator", ["ising", "fermion_operator"])
@pytest.mark.parametrize("chunk_size", [None, 4])
@pytest.mark.parametrize("param_dtype", [float, complex])
def test_packed_kernel_zero_samples(operator, chunk_size, param_dtype):
    # Without samples, the local values are an empty array of the dtype of
    # those of the padded kernel, and their derivatives are zero. (The padded kernel
    # itself fails without samples for the fermionic operator: the dtype is
    # that of its local values of one sample.)
    if operator == "ising":
        hi = nk.hilbert.Spin(0.5, 6)
        H = nk.operator.IsingJax(hi, nk.graph.Chain(6), h=1.0)
    else:
        hi, H, _ = _hubbard(False)
    # Set the operator up outside of any trace.
    jax.block_until_ready(H.get_conn_padded(hi.random_state(jax.random.key(0), 1)))
    ma = nk.models.RBM(alpha=1, param_dtype=param_dtype)
    σ = jnp.zeros((0, hi.size), dtype=hi.random_state(jax.random.key(0), 1).dtype)
    variables = ma.init(jax.random.key(1), jnp.zeros((1, hi.size)))

    kernel = _packed(chunk_size)
    expected = jax.eval_shape(
        lambda v, σ: kernels.local_value_kernel_jax(ma.apply, v, σ, H),
        variables,
        jnp.zeros((1, hi.size), dtype=σ.dtype),
    )
    result = jax.jit(lambda v, σ: kernel(ma.apply, v, σ, H))(variables, σ)
    assert result.shape == (0,)
    assert result.dtype == expected.dtype

    def loss(v):
        return jnp.sum(jnp.abs(kernel(ma.apply, v, σ, H)) ** 2)

    grad = jax.jit(jax.grad(loss))(variables)
    jax.tree.map(lambda g: np.testing.assert_array_equal(g, 0), grad)


def _bitwise(x):
    x = np.asarray(x)
    return x.dtype, x.shape, x.tobytes()


@pytest.mark.parametrize("chunk_size", [None, 1024])
def test_packed_kernel_bitwise_repeatable(chunk_size):
    # Repeated evaluations on the same inputs must give the same bits. A
    # scatter-add accumulating several terms per sample would not guarantee
    # it on GPU, where the order of the overlapping updates is not fixed: many
    # samples, so that there are many of them.
    vs, H = _vstate("heisenberg_triangular")
    σ = vs.hilbert.random_state(jax.random.key(0), 4096 * jax.device_count())
    kernel = _packed(chunk_size)
    f = jax.jit(lambda v, σ, H: kernel(vs._apply_fun, v, σ, H))
    expected = _bitwise(f(vs.variables, σ, H))
    for _ in range(20):
        assert _bitwise(f(vs.variables, σ, H)) == expected


@pytest.mark.parametrize("chunk_size", [None, 32])
def test_packed_kernel_gradient_bitwise_repeatable(chunk_size):
    vs, H = _vstate("hubbard")
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    kernel = _packed(chunk_size)

    def loss(params, H):
        variables = {**vs.variables, "params": params}
        return jnp.sum(jnp.abs(kernel(vs._apply_fun, variables, σ, H)) ** 2)

    f = jax.jit(jax.grad(loss))
    expected = jax.tree.map(_bitwise, f(vs.parameters, H))
    for _ in range(10):
        assert jax.tree.map(_bitwise, f(vs.parameters, H)) == expected


_REPEATABILITY_SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np
    import jax
    import flax
    import netket as nk
    from netket.vqs.mc import kernels_packed

    inputs, output = sys.argv[1:]
    data = np.load(inputs)
    g = nk.graph.Triangular([3, 4], pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    H = nk.operator.Heisenberg(hi, g)
    model = nk.models.RBM(alpha=2, param_dtype=complex)
    variables = flax.serialization.msgpack_restore(data["variables"].tobytes())

    out = {}
    for chunk_size in (None, 32):
        f = jax.jit(
            lambda v, σ, H: kernels_packed.local_value_kernel_jax_packed(
                model.apply, v, σ, H, chunk_size=chunk_size
            )
        )
        out[str(chunk_size)] = np.asarray(f(variables, data["samples"], H))
    np.savez(output, **out)
    """
)


@pytest.mark.skipif(jax.process_count() > 1, reason="spawns single-process runs")
def test_packed_kernel_bitwise_repeatable_across_processes(tmp_path):
    # Fresh processes on the same hardware and software, given the same saved
    # inputs, must give the same bits.
    vs, H = _vstate("heisenberg_triangular")
    inputs = tmp_path / "inputs.npz"
    np.savez(
        inputs,
        samples=np.asarray(vs.samples.reshape(-1, vs.hilbert.size)),
        variables=np.frombuffer(
            flax.serialization.msgpack_serialize(jax.device_get(vs.variables)),
            dtype=np.uint8,
        ),
    )
    # On GPU, XLA's autotuner can choose a different implementation of the
    # network (e.g. of a matrix product) in every process, which changes the
    # rounding of log psi with any kernel: it is disabled, to test the kernel.
    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0",
    }
    outputs = []
    for k in range(2):
        output = tmp_path / f"output{k}.npz"
        subprocess.run(
            [sys.executable, "-c", _REPEATABILITY_SCRIPT, str(inputs), str(output)],
            env=env,
            check=True,
        )
        outputs.append(dict(np.load(output)))
    assert set(outputs[0]) == {"None", "32"}
    for key, value in outputs[0].items():
        assert _bitwise(outputs[1][key]) == _bitwise(value)


## The primitive


def _rows_function(pars, x):
    # A row-wise function with a pytree output.
    a, b = pars
    return {"sum": jnp.sum(jnp.sin(a * x), axis=-1), "vec": jnp.cos(x[:, :3] * b)}


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("chunk_size", [None, 1, 5, 64])
def test_apply_packed(fraction, chunk_size):
    n, d = 37, 4
    x = jax.random.normal(jax.random.key(0), (n, d))
    where = jax.random.uniform(jax.random.key(1), (n,)) < fraction
    pars = (jnp.array(1.3), jnp.array(0.7))
    fill = {"sum": jnp.array(-1.0), "vec": jnp.full((n, 1), 2.0)}

    # On the arrays as given: 37 rows are not split among devices.
    out = jax.jit(
        lambda pars, x, where, fill: nk.jax.apply_packed(
            _rows_function,
            pars,
            x,
            where,
            fill,
            chunk_size=chunk_size,
            axis_0_is_sharded=False,
        )
    )(pars, x, where, fill)

    ref = _rows_function(pars, x)
    np.testing.assert_allclose(out["sum"], jnp.where(where, ref["sum"], -1.0))
    np.testing.assert_allclose(
        out["vec"], jnp.where(where[:, None], ref["vec"], 2.0), rtol=1e-13
    )
    assert out["vec"].shape == (n, 3)


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_apply_packed_slots(chunk_size):
    # With a two-dimensional mask, f is evaluated on pairs of a row and a slot.
    n, m, d = 9, 5, 3
    x = jax.random.normal(jax.random.key(0), (n, d))
    where = jax.random.uniform(jax.random.key(1), (n, m)) < 0.5

    def f(pars, x, slots):
        return pars * jnp.sum(x, axis=-1) + slots

    out = nk.jax.apply_packed(
        f, 2.0, x, where, 0.0, chunk_size=chunk_size, axis_0_is_sharded=False
    )
    expected = 2.0 * jnp.sum(x, axis=-1)[:, None] + jnp.arange(m)
    np.testing.assert_allclose(
        out, jnp.where(where, expected, 0.0), rtol=1e-13, atol=1e-13
    )

    # An empty slot axis.
    out = nk.jax.apply_packed(
        f, 2.0, x, where[:, :0], 0.0, chunk_size=chunk_size, axis_0_is_sharded=False
    )
    assert out.shape == (n, 0)


@pytest.mark.parametrize("x64", [True, False])
def test_apply_packed_integer_slots(x64):
    # The slots have the same dtype when the shape of the output is traced and
    # when f is evaluated: that of the indices of jnp.nonzero, int64 with
    # 64-bit types and int32 without.
    with jax.enable_x64(x64):
        index_dtype = jnp.nonzero(jnp.ones(1))[0].dtype
        assert index_dtype == (jnp.int64 if x64 else jnp.int32)

        out = nk.jax.apply_packed(
            lambda p, x, slots: slots,
            None,
            jnp.ones((2, 1)),
            jnp.array([[True, False, True], [False, True, False]]),
            0,
            chunk_size=2,
            axis_0_is_sharded=False,
        )
        assert out.dtype == index_dtype
        np.testing.assert_array_equal(out, [[0, 0, 2], [0, 1, 0]])

        # Integer and floating outputs, with integer rows, under jit and vmap.
        n, m = 5, 3
        x = jnp.arange(2 * n, dtype=jnp.int32).reshape(n, 2)
        where = jax.random.uniform(jax.random.key(0), (2, n, m)) < 0.5

        def f(p, x, slots):
            return {"slot": slots, "row": x[:, 0], "value": p * x[:, 1] + slots}

        fill = {"slot": -1, "row": -2, "value": 0.5}

        def packed(where, x=x):
            return nk.jax.apply_packed(
                f, 2.0, x, where, fill, chunk_size=2, axis_0_is_sharded=False
            )

        float_dtype = jnp.float64 if x64 else jnp.float32
        x_np, slots = np.asarray(x), np.arange(m)
        values = {
            "slot": np.broadcast_to(slots, (n, m)),
            "row": np.broadcast_to(x_np[:, :1], (n, m)),
            "value": 2.0 * x_np[:, 1:] + slots,
        }
        expected = [
            {k: np.where(w, v, fill[k]) for k, v in values.items()} for w in where
        ]
        results = [jax.jit(packed)(where[0]), jax.jit(jax.vmap(packed))(where)]
        for result, expected_i in [(results[0], expected[0])] + [
            (jax.tree.map(lambda o: o[i], results[1]), expected[i]) for i in range(2)
        ]:
            assert result["slot"].dtype == index_dtype
            assert result["row"].dtype == jnp.int32
            assert result["value"].dtype == float_dtype
            jax.tree.map(np.testing.assert_array_equal, result, expected_i)

        # An empty slot axis.
        out = packed(where[0, :, :0])
        assert jax.tree.map(lambda o: (o.shape, o.dtype), out) == {
            "slot": ((n, 0), index_dtype),
            "row": ((n, 0), jnp.int32),
            "value": ((n, 0), float_dtype),
        }


def _sqrt_rows(p, x):
    # Not differentiable on the rows with x < 0, which are not selected.
    return jnp.sqrt(p * x[:, 0])


def _sqrt_slots(p, x, slots):
    return jnp.sqrt(p * x[:, 0] * (slots + 1))


def test_apply_packed_padding_gradient():
    # The padding of the last chunk repeats a selected row: the derivative of
    # its discarded value is zero, not NaN, also when f is not differentiable
    # on the rows that are not selected.
    x = jnp.array([[1.0], [-1.0]])
    where = jnp.array([True, False])

    def loss(p, minimum):
        return nk.jax.apply_packed(
            lambda p, x: jnp.sqrt(p * x[:, 0]),
            p,
            x,
            where,
            0.0,
            chunk_size=2,
            min_chunk_size=minimum,
            axis_0_is_sharded=False,
        ).sum()

    for minimum in (1, 2):
        value, grad = jax.jit(jax.value_and_grad(lambda p: loss(p, minimum)))(1.0)
        np.testing.assert_allclose(value, 1.0)
        np.testing.assert_allclose(grad, 0.5)

    # The same with pairs of a row and a slot.
    def loss_slots(p, minimum):
        return nk.jax.apply_packed(
            _sqrt_slots,
            p,
            x,
            jnp.array([[True, False], [False, False]]),
            0.0,
            chunk_size=2,
            min_chunk_size=minimum,
            axis_0_is_sharded=False,
        ).sum()

    for minimum in (1, 2):
        value, grad = jax.jit(jax.value_and_grad(lambda p: loss_slots(p, minimum)))(1.0)
        np.testing.assert_allclose(value, 1.0)
        np.testing.assert_allclose(grad, 0.5)


@pytest.mark.parametrize("slots", [False, True])
def test_apply_packed_empty_element_derivatives(slots):
    # Under vmap of a derivative, every element of the batch evaluates the
    # chunks of the largest one: an element without selected rows evaluates
    # them on rows where f is not differentiable, whose derivatives must not
    # reach the result.
    x = jnp.array([[[1.0], [4.0]], [[-1.0], [-4.0]]])
    if slots:
        f = _sqrt_slots
        where = jnp.array([[[True, True], [True, True]], [[False] * 2] * 2])
    else:
        f = _sqrt_rows
        where = jnp.array([[True, True], [False, False]])

    def loss(p, x, where):
        return nk.jax.apply_packed(
            f,
            p,
            x,
            where,
            0.0,
            chunk_size=2,
            min_chunk_size=1,
            axis_0_is_sharded=False,
        ).sum()

    expected = jnp.stack([jax.grad(loss)(1.0, x[i], where[i]) for i in range(2)])
    assert expected[1] == 0
    vmap_grad = jax.jit(jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)))
    np.testing.assert_allclose(vmap_grad(1.0, x, where), expected)

    def grad_vmap(p, x, where):
        return jax.grad(lambda p: jax.vmap(loss, (None, 0, 0))(p, x, where).sum())(p)

    np.testing.assert_allclose(jax.jit(grad_vmap)(1.0, x, where), expected.sum())

    def jvp(p, x, where):
        return jax.jvp(lambda p: loss(p, x, where), (p,), (1.0,))[1]

    vmap_jvp = jax.jit(jax.vmap(jvp, in_axes=(None, 0, 0)))
    np.testing.assert_allclose(vmap_jvp(1.0, x, where), expected)

    hessian = jnp.stack([jax.hessian(loss)(1.0, x[i], where[i]) for i in range(2)])
    vmap_hessian = jax.jit(jax.vmap(jax.hessian(loss), in_axes=(None, 0, 0)))
    np.testing.assert_allclose(vmap_hessian(1.0, x, where), hessian)
    assert hessian[1] == 0


def _shard(x, axis=0):
    return jax.jit(lambda x: nk.jax.sharding.shard_along_axis(x, axis))(x)


@pytest.mark.parametrize("slots", [False, True])
def test_apply_packed_sharded(slots):
    # With axis_0_is_sharded, the rows of every device are packed separately:
    # the result is that of the arrays packed at once, the output keeps the
    # sharding of the rows, and there is no communication.
    n, d, m = 16 * jax.device_count(), 4, 3
    x = jax.random.normal(jax.random.key(0), (n, d))
    shape = (n, m) if slots else (n,)
    where = jax.random.uniform(jax.random.key(1), shape) < 0.4
    pars = (jnp.array(1.3), jnp.array(0.7))
    if slots:

        def f(pars, x, slots):
            out = _rows_function(pars, x)
            return {"sum": out["sum"] + slots, "vec": out["vec"] * slots[:, None]}

        # one value per row, and one per slot (replicated)
        fill = {"sum": jnp.linspace(-1, 1, n)[:, None], "vec": jnp.full((3,), 2.0)}
    else:
        f = _rows_function
        fill = {"sum": jnp.linspace(-1, 1, n), "vec": jnp.array(2.0)}

    def packed(pars, x, where, fill, sharded):
        return nk.jax.apply_packed(
            f, pars, x, where, fill, chunk_size=5, axis_0_is_sharded=sharded
        )

    expected = jax.jit(partial(packed, sharded=False))(pars, x, where, fill)
    xs, ws = _shard(x), _shard(where)
    fs = {"sum": _shard(fill["sum"]), "vec": fill["vec"]}
    g = jax.jit(partial(packed, sharded=True))
    result = g(pars, xs, ws, fs)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14),
        result,
        expected,
    )
    if jax.device_count() > 1 and nk.config.netket_sharding:
        for leaf in jax.tree.leaves(result):
            assert leaf.sharding.spec[0] == "S"
    hlo = g.lower(pars, xs, ws, fs).compile().as_text()
    for collective in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
        assert collective not in hlo

    # Under vmap over the rows, the custom batching rule still applies on every
    # device.
    xb = _shard(jnp.stack([x, x[::-1]]), 1)
    wb = _shard(jnp.stack([where, where[::-1]]), 1)
    batched = jax.vmap(partial(packed, sharded=True), in_axes=(None, 0, 0, None))
    result = jax.jit(batched)(pars, xb, wb, fs)
    for i, (xi, wi) in enumerate([(x, where), (x[::-1], where[::-1])]):
        expected_i = jax.jit(partial(packed, sharded=False))(pars, xi, wi, fill)
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a[i], b, rtol=1e-14, atol=1e-14),
            result,
            expected_i,
        )
    assert _count(jax.make_jaxpr(batched)(pars, xb, wb, fs).jaxpr, "cond") > 0

    # Forward and reverse derivatives with respect to pars and fill.
    def loss(pars, fill, x, where, sharded):
        out = packed(pars, x, where, fill, sharded)
        return sum(jnp.sum(jnp.sin(o)) for o in jax.tree.leaves(out))

    grad = jax.jit(jax.grad(loss, argnums=(0, 1)), static_argnums=4)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13),
        grad(pars, fs, xs, ws, True),
        grad(pars, fill, x, where, False),
    )
    tangent = (jnp.array(0.3), jnp.array(-0.2))

    def jvp(x, where, fill, sharded):
        return jax.jvp(lambda p: loss(p, fill, x, where, sharded), (pars,), (tangent,))

    jvp = jax.jit(jvp, static_argnums=3)
    np.testing.assert_allclose(
        jvp(xs, ws, fs, True), jvp(x, where, fill, False), rtol=1e-12, atol=1e-13
    )


## Derivatives


def _loss_fns(system, param_dtype, chunk_size):
    vs, H = _vstate(system, param_dtype=param_dtype)
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    jax.block_until_ready(H.get_conn_padded(σ[:1]))
    p0, unravel = ravel_pytree(vs.parameters)

    def eloc(kernel, p, H):
        return kernel(vs._apply_fun, {**vs.variables, "params": unravel(p)}, σ, H)

    def loss(kernel, p, H):
        return jnp.sum(jnp.abs(eloc(kernel, p, H)) ** 2)

    t = jax.random.normal(jax.random.key(1), p0.shape, dtype=p0.dtype)
    return eloc, loss, p0, t, H


@pytest.mark.parametrize("system", ["heisenberg_triangular", "hubbard"])
@pytest.mark.parametrize("param_dtype", [float, complex])
def test_packed_kernel_higher_order_derivatives(system, param_dtype):
    eloc, loss, p0, t, H = _loss_fns(system, param_dtype, 16)
    packed, padded = _packed(16), kernels.local_value_kernel_jax

    def hvp(kernel, p, t, H):
        # forward over reverse
        return jax.jvp(jax.grad(lambda p: loss(kernel, p, H)), (p,), (t,))[1]

    def rev_rev(kernel, p, t, H):
        grad = jax.grad(lambda p: loss(kernel, p, H))
        return jax.grad(lambda p: jnp.real(jnp.vdot(grad(p), t)))(p)

    def third(kernel, p, t, H):
        return jax.jvp(
            lambda p: jnp.real(jnp.vdot(hvp(kernel, p, t, H), t)), (p,), (t,)
        )[1]

    for derivative in (hvp, rev_rev, third):
        f = jax.jit(derivative, static_argnums=0)
        _assert_close(f(packed, p0, t, H), f(padded, p0, t, H), 1e-12)
    if param_dtype is float:
        f = jax.jit(jax.hessian(loss, argnums=1), static_argnums=0)
        _assert_close(f(packed, p0, H), f(padded, p0, H), 1e-12)
        f = jax.jit(jax.jacrev(eloc, argnums=1), static_argnums=0)
        _assert_close(f(packed, p0, H), f(padded, p0, H), 1e-12)
        f = jax.jit(jax.jacfwd(eloc, argnums=1), static_argnums=0)
        _assert_close(f(packed, p0, H), f(padded, p0, H), 1e-12)


@pytest.mark.parametrize("pnc", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_packed_kernel_fermions(pnc, chunk_size):
    # Fermionic operators (FermionOperator2ndJax and the particle-number
    # conserving FermiHubbardJax) go through the generic path, with their own
    # sign conventions in the matrix elements: local values and gradients.
    vs, H = _vstate("hubbard_pnc" if pnc else "hubbard")
    np.testing.assert_allclose(
        _local_values(vs, H, chunk_size, packed=True),
        _local_values(vs, H, None, packed=False),
        rtol=1e-12,
        atol=1e-12,
    )

    σ = vs.samples.reshape(-1, vs.hilbert.size)

    def loss(kernel, params, H):
        variables = {**vs.variables, "params": params}
        return jnp.sum(jnp.abs(kernel(vs._apply_fun, variables, σ, H)) ** 2)

    f = jax.jit(jax.grad(loss, argnums=1), static_argnums=0)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12),
        f(_packed(chunk_size), vs.parameters, H),
        f(kernels.local_value_kernel_jax, vs.parameters, H),
    )


## vmap


def _count(jaxpr, name):
    n = 0
    for eqn in jaxpr.eqns:
        n += eqn.primitive.name == name
        for param in eqn.params.values():
            for p in param if isinstance(param, tuple | list) else (param,):
                if isinstance(p, jax.extend.core.ClosedJaxpr):
                    n += _count(p.jaxpr, name)
                elif isinstance(p, jax.extend.core.Jaxpr):
                    n += _count(p, name)
    return n


@pytest.mark.parametrize("system", ["heisenberg_triangular", "hubbard"])
@pytest.mark.parametrize("chunk_size", [None, 16])
def test_packed_kernel_vmap(system, chunk_size):
    # Under vmap over the samples, the batch is packed at once: the chunks
    # are still selected with lax.cond.
    vs, H = _vstate(system)
    σ = vs.samples.reshape(4, -1, vs.hilbert.size)
    kernel = _packed(chunk_size)

    def eloc(variables, σ):
        return kernel(vs._apply_fun, variables, σ, H)

    expected = jnp.stack([eloc(vs.variables, s) for s in σ])
    batched = jax.vmap(eloc, in_axes=(None, 0))
    np.testing.assert_allclose(
        jax.jit(batched)(vs.variables, σ), expected, rtol=1e-12, atol=1e-12
    )
    assert _count(jax.make_jaxpr(batched)(vs.variables, σ).jaxpr, "cond") > 0

    # Nested vmap.
    nested = jax.vmap(jax.vmap(eloc, in_axes=(None, 0)), in_axes=(None, 0))
    np.testing.assert_allclose(
        jax.jit(nested)(vs.variables, σ.reshape(2, 2, *σ.shape[1:])).reshape(4, -1),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )

    # vmap over the parameters keeps the packing of the samples.
    variables = jax.tree.map(lambda p: jnp.stack([p, 1.1 * p]), vs.variables)
    expected = jnp.stack(
        [eloc(jax.tree.map(lambda p: p[i], variables), σ[0]) for i in range(2)]
    )
    batched = jax.vmap(eloc, in_axes=(0, None))
    np.testing.assert_allclose(
        jax.jit(batched)(variables, σ[0]), expected, rtol=1e-12, atol=1e-12
    )
    assert _count(jax.make_jaxpr(batched)(variables, σ[0]).jaxpr, "cond") > 0

    # Both batched.
    expected = jnp.stack(
        [eloc(jax.tree.map(lambda p: p[i], variables), σ[i]) for i in range(2)]
    )
    batched = jax.vmap(eloc, in_axes=(0, 0))
    np.testing.assert_allclose(
        jax.jit(batched)(variables, σ[:2]), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("system", ["heisenberg_triangular", "hubbard"])
def test_packed_kernel_vmap_of_derivatives(system):
    # Derivatives of a batch, and batches of derivatives (where every chunk is
    # evaluated).
    vs, H = _vstate(system)
    σ = vs.samples.reshape(4, -1, vs.hilbert.size)
    p0, unravel = ravel_pytree(vs.parameters)
    t = jnp.ones_like(p0)

    def loss(kernel, p, σ):
        variables = {**vs.variables, "params": unravel(p)}
        return jnp.sum(jnp.abs(kernel(vs._apply_fun, variables, σ, H)) ** 2)

    def vmap_grad(kernel, p, σ):
        return jax.vmap(jax.grad(lambda p, σ: loss(kernel, p, σ)), (None, 0))(p, σ)

    def grad_vmap(kernel, p, σ):
        return jax.grad(lambda p: jnp.sum(jax.vmap(lambda σ: loss(kernel, p, σ))(σ)))(p)

    def vmap_jvp(kernel, p, σ):
        def jvp(σ):
            return jax.jvp(lambda p: loss(kernel, p, σ), (p,), (t,))[1]

        return jax.vmap(jvp)(σ)

    for transform in (vmap_grad, grad_vmap, vmap_jvp):
        f = jax.jit(transform, static_argnums=0)
        packed = _packed(16)
        _assert_close(f(packed, p0, σ), f(kernels.local_value_kernel_jax, p0, σ), 1e-12)
        # The chunks are still selected by lax.cond.
        jaxpr = jax.make_jaxpr(transform, static_argnums=0)(packed, p0, σ)
        assert _count(jaxpr.jaxpr, "cond") > 0
