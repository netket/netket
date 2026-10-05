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

"""Tests for the flattened local-value kernel of jax operators."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import flax

import netket as nk
from netket.vqs.mc import kernels

from .. import common


def _heisenberg_chain():
    g = nk.graph.Chain(12, pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    return hi, nk.operator.Heisenberg(hi, g), nk.sampler.MetropolisExchange(hi, graph=g)


def _heisenberg_triangular():
    g = nk.graph.Triangular([3, 4], pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes, total_sz=0)
    return hi, nk.operator.Heisenberg(hi, g), nk.sampler.MetropolisExchange(hi, graph=g)


def _ising():
    g = nk.graph.Chain(10, pbc=True)
    hi = nk.hilbert.Spin(0.5, g.n_nodes)
    return hi, nk.operator.IsingJax(hi, g, h=1.0), nk.sampler.MetropolisLocal(hi)


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
    "ising": _ising,
    "hubbard": lambda: _hubbard(False),
    "hubbard_pnc": lambda: _hubbard(True),
}


def _vstate(system):
    hi, H, sa = SYSTEMS[system]()
    sa = sa.replace(n_chains=8 * jax.device_count())
    ma = nk.models.RBM(alpha=2, param_dtype=complex)
    vs = nk.vqs.MCState(sa, ma, n_samples=64 * jax.device_count(), seed=0)
    return vs, H


def _local_estimators(vs, H, chunk_size, flattened):
    old = nk.config.netket_experimental_flattened_kernel
    nk.config.netket_experimental_flattened_kernel = flattened
    try:
        return np.asarray(vs.local_estimators(H, chunk_size=chunk_size).data)
    finally:
        nk.config.netket_experimental_flattened_kernel = old


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("chunk_size", [None, 7, 64, 100000])
def test_flattened_kernel_matches_padded(system, chunk_size):
    vs, H = _vstate(system)
    expected = _local_estimators(vs, H, None, flattened=False)
    result = _local_estimators(vs, H, chunk_size, flattened=True)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_flattened_kernel_dispatch():
    vs, H = _vstate("heisenberg_chain")
    old = nk.config.netket_experimental_flattened_kernel
    try:
        nk.config.netket_experimental_flattened_kernel = False
        assert nk.vqs.get_local_kernel(vs, H) is kernels.local_value_kernel_jax
        assert (
            nk.vqs.get_local_kernel(vs, H, 16) is kernels.local_value_kernel_jax_chunked
        )
        nk.config.netket_experimental_flattened_kernel = True
        assert (
            nk.vqs.get_local_kernel(vs, H) is kernels.local_value_kernel_jax_flattened
        )
        assert (
            nk.vqs.get_local_kernel(vs, H, 16)
            is kernels.local_value_kernel_jax_flattened
        )
    finally:
        nk.config.netket_experimental_flattened_kernel = old


@pytest.mark.parametrize("chunk_size", [None, 32])
def test_flattened_kernel_no_recompilation(chunk_size):
    # The number of nonzero connected elements changes from batch to batch,
    # but the kernel must be compiled only once.
    vs, H = _vstate("heisenberg_triangular")
    n_traces = 0

    def logpsi(variables, x):
        nonlocal n_traces
        n_traces += 1
        return vs._apply_fun(variables, x)

    @jax.jit
    def eloc(variables, σ, H):
        return kernels.local_value_kernel_jax_flattened(
            logpsi, variables, σ, H, chunk_size=chunk_size
        )

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


@pytest.mark.parametrize("chunk_size", [None, 32])
def test_flattened_kernel_gradient(chunk_size):
    # The while loop is not reverse-differentiable: the custom vjp must give
    # the gradient of the padded kernel.
    vs, H = _vstate("hubbard")
    σ = vs.samples.reshape(-1, vs.hilbert.size)

    # The operator is passed as an argument: FermionOperator2ndJax must not be
    # set up inside of a trace.
    def loss(kernel, params, H):
        variables = {**vs.variables, "params": params}
        return jnp.sum(jnp.abs(kernel(vs._apply_fun, variables, σ, H)) ** 2)

    def flattened(*args):
        return kernels.local_value_kernel_jax_flattened(*args, chunk_size=chunk_size)

    expected = jax.jit(
        jax.grad(lambda p, H: loss(kernels.local_value_kernel_jax, p, H))
    )(vs.parameters, H)
    result = jax.jit(jax.grad(lambda p, H: loss(flattened, p, H)))(vs.parameters, H)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12),
        result,
        expected,
    )


def test_flattened_kernel_expect_and_grad_nonhermitian():
    vs, H = _vstate("heisenberg_chain")
    old_flat = nk.config.netket_experimental_flattened_kernel
    old_exp = nk.config.netket_experimental
    try:
        nk.config.netket_experimental = True
        nk.config.netket_experimental_flattened_kernel = False
        E0, g0 = vs.expect_and_grad(H, use_covariance=False)
        nk.config.netket_experimental_flattened_kernel = True
        E1, g1 = vs.expect_and_grad(H, use_covariance=False)
    finally:
        nk.config.netket_experimental_flattened_kernel = old_flat
        nk.config.netket_experimental = old_exp
    np.testing.assert_allclose(E1.mean, E0.mean, rtol=1e-12)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12), g1, g0
    )


@common.onlyif_sharding_single_process
def test_flattened_kernel_no_collectives():
    # The compaction must run per device: under GSPMD it would gather the
    # connected configurations of all devices.
    vs, H = _vstate("heisenberg_triangular")
    σ = vs.samples.reshape(-1, vs.hilbert.size)

    f = jax.jit(
        lambda v, σ, H: kernels.local_value_kernel_jax_flattened(vs._apply_fun, v, σ, H)
    )
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
def test_flattened_kernel_coefficient_gradient_at_zero(chunk_size):
    # At h = 0 every off-diagonal matrix element of the Ising model is zero,
    # but its derivative with respect to h is not: the kernel must not drop
    # the connected configurations based on the value of their matrix element.
    vs, _ = _vstate("ising")
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    graph = nk.graph.Chain(vs.hilbert.size, pbc=True)

    def flattened(*args):
        return kernels.local_value_kernel_jax_flattened(*args, chunk_size=chunk_size)

    def loss(kernel, h):
        H = nk.operator.IsingJax(vs.hilbert, graph, h=h, J=1.0)
        return jnp.sum(kernel(vs._apply_fun, vs.variables, σ, H)).real

    h = jnp.array(0.0)
    expected = jax.jit(jax.grad(lambda h: loss(kernels.local_value_kernel_jax, h)))(h)
    result = jax.jit(jax.grad(lambda h: loss(flattened, h)))(h)
    eps = 1e-5
    finite_diff = (loss(flattened, h + eps) - loss(flattened, h - eps)) / (2 * eps)

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
def test_flattened_kernel_empty_operator(operator, chunk_size):
    # An operator without connected elements is a valid zero operator.
    vs, H = _empty_operator(operator)
    assert H.max_conn_size == 0
    expected = _local_estimators(vs, H, chunk_size, flattened=False)
    result = _local_estimators(vs, H, chunk_size, flattened=True)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, 0)
    np.testing.assert_array_equal(expected, 0)

    σ = vs.samples.reshape(-1, vs.hilbert.size)

    def loss(params, H):
        variables = {**vs.variables, "params": params}
        out = kernels.local_value_kernel_jax_flattened(
            vs._apply_fun, variables, σ, H, chunk_size=chunk_size
        )
        return jnp.sum(jnp.abs(out) ** 2)

    grad = jax.jit(jax.grad(loss))(vs.parameters, H)
    jax.tree.map(lambda g: np.testing.assert_array_equal(g, 0), grad)


def _bitwise(x):
    x = np.asarray(x)
    return x.dtype, x.shape, x.tobytes()


@pytest.mark.parametrize("chunk_size", [None, 32])
def test_flattened_kernel_bitwise_repeatable(chunk_size):
    # Repeated evaluations on the same inputs must give the same bits. A
    # scatter-add accumulating several terms per sample would not guarantee
    # it on GPU, where the order of the overlapping updates is not fixed.
    vs, H = _vstate("hubbard")
    σ = vs.samples.reshape(-1, vs.hilbert.size)
    f = jax.jit(
        lambda v, σ, H: kernels.local_value_kernel_jax_flattened(
            vs._apply_fun, v, σ, H, chunk_size=chunk_size
        )
    )
    expected = _bitwise(f(vs.variables, σ, H))
    for _ in range(20):
        assert _bitwise(f(vs.variables, σ, H)) == expected


@pytest.mark.parametrize("chunk_size", [None, 32])
def test_flattened_kernel_gradient_bitwise_repeatable(chunk_size):
    vs, H = _vstate("hubbard")
    σ = vs.samples.reshape(-1, vs.hilbert.size)

    def loss(params, H):
        variables = {**vs.variables, "params": params}
        out = kernels.local_value_kernel_jax_flattened(
            vs._apply_fun, variables, σ, H, chunk_size=chunk_size
        )
        return jnp.sum(jnp.abs(out) ** 2)

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
    from netket.vqs.mc import kernels

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
            lambda v, σ, H: kernels.local_value_kernel_jax_flattened(
                model.apply, v, σ, H, chunk_size=chunk_size
            )
        )
        out[str(chunk_size)] = np.asarray(f(variables, data["samples"], H))
    np.savez(output, **out)
    """
)


@pytest.mark.skipif(jax.process_count() > 1, reason="spawns single-process runs")
def test_flattened_kernel_bitwise_repeatable_across_processes(tmp_path):
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
