# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Deduplicated model evaluation, contribution preservation and deterministic AD."""

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import netket as nk
from netket.vqs.mc import kernels


def logpsi(p, x):
    return p * jnp.sum(x * jnp.arange(1, x.shape[-1] + 1), axis=-1) + 0.03j * x[..., 0]


@jax.tree_util.register_pytree_node_class
class RepeatedConnections(nk.operator.DiscreteJaxOperator):
    def __init__(self, op):
        super().__init__(op.hilbert)
        self.op = op

    @property
    def dtype(self):
        return self.op.dtype

    @property
    def max_conn_size(self):
        return 3 * self.op.max_conn_size

    def tree_flatten(self):
        return (self.op,), None

    @classmethod
    def tree_unflatten(cls, metadata, leaves):
        return cls(leaves[0])

    def get_conn_padded(self, x):
        xp, mels = self.op.get_conn_padded(x)
        return (
            jnp.concatenate((xp, xp, xp), axis=1),
            jnp.concatenate((2 * mels, -0.3 * mels, -0.7 * mels), axis=1),
        )


@pytest.mark.parametrize("chunk_size", [1, 3, 16, None])
def test_exact_unique_model_work_and_no_recompilation(chunk_size):
    hi = nk.hilbert.Spin(0.5, 4)
    H = RepeatedConnections(nk.operator.IsingJax(hi, nk.graph.Chain(4), h=0.7))
    observed, traces = [], []

    def counted(p, x):
        traces.append(x.shape)
        jax.debug.callback(lambda x: observed.extend(map(tuple, np.asarray(x))), x)
        return logpsi(p, x)

    run = jax.jit(
        lambda p, x, H: kernels.local_value_kernel_jax_unique(
            counted, p, x, H, chunk_size=chunk_size, min_chunk_size=1
        )
    )
    counts = []
    for repeated in (True, False):
        x = jnp.asarray(hi.all_states())
        if repeated:
            x = jnp.repeat(x[:1], len(x), axis=0)
        x = nk.jax.sharding.shard_along_axis(x, axis=0)
        observed.clear()
        actual = jax.block_until_ready(run(0.17, x, H))
        jax.effects_barrier()
        expected = kernels.local_value_kernel_jax(logpsi, 0.17, x, H)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        xp, _ = H.get_conn_padded(x)
        expected_rows = []
        shards = jax.device_count() if nk.config.netket_sharding else 1
        for local_x, local_xp in zip(
            np.split(np.asarray(x), shards), np.split(np.asarray(xp), shards)
        ):
            rows = np.concatenate((local_x, local_xp.reshape(-1, hi.size)))
            expected_rows.extend(map(tuple, np.unique(rows, axis=0)))
        assert Counter(observed) == Counter(expected_rows)
        counts.append(len(observed))
        if repeated:
            initial_traces = len(traces)
        assert len(traces) == initial_traces
    assert counts[0] < counts[1]


@pytest.mark.parametrize("chunk_size", [3, 16])
def test_operator_model_and_mixed_derivatives(chunk_size):
    hi = nk.hilbert.Spin(0.5, 4)
    x = nk.jax.sharding.shard_along_axis(jnp.asarray(hi.all_states()), axis=0)

    def loss(p, kernel):
        H = RepeatedConnections(nk.operator.IsingJax(hi, nk.graph.Chain(4), h=p[1]))
        values = kernel(logpsi, p[0], x, H)
        return jnp.sum(jnp.abs(values) ** 2)

    def unique(*args):
        return kernels.local_value_kernel_jax_unique(*args, chunk_size=chunk_size)

    p = jnp.array([0.17, 0.0])
    tangent = jnp.array([0.3, 0.8])
    for transform in (
        lambda f: jax.jit(lambda p: jax.jvp(f, (p,), (tangent,))),
        lambda f: jax.jit(jax.grad(f)),
        lambda f: jax.jit(jax.jacfwd(jax.jacrev(f))),
    ):
        expected = transform(lambda p: loss(p, kernels.local_value_kernel_jax))(p)
        actual = transform(lambda p: loss(p, unique))(p)
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-10),
            actual,
            expected,
        )


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_expansion_and_its_transpose(dtype):
    rows = jnp.array([[1, 2], [1, 2], [-1, 3], [1, 2], [-1, 3], [4, 5]])
    order, starts, positions, representatives, count = kernels._unique_row_plan(rows)
    reference_rows, inverse = np.unique(np.asarray(rows), axis=0, return_inverse=True)
    values = jnp.arange(6, dtype=jnp.float64).astype(dtype) + 0.2
    if dtype == jnp.complex128:
        values = values + 0.7j * values
    f = lambda v: kernels._expand_unique_rows(v, order, starts, positions)
    np.testing.assert_array_equal(f(values), np.asarray(values)[inverse])
    ct = jnp.asarray([0.2, -0.7, 1.5, 0.1, -0.3, 4.0], dtype=dtype)
    actual = jax.jit(lambda v, ct: jax.vjp(f, v)[1](ct)[0])(values, ct)
    expected = np.zeros_like(values)
    for i, c in zip(inverse, np.asarray(ct)):
        expected[i] += c
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert int(count) == len(reference_rows)
    np.testing.assert_array_equal(
        np.asarray(rows)[np.asarray(representatives)[: int(count)]], reference_rows
    )


def test_unique_no_collectives():
    hi = nk.hilbert.Spin(0.5, 4)
    H = nk.operator.Heisenberg(hi, nk.graph.Chain(4))
    x = nk.jax.sharding.shard_along_axis(jnp.asarray(hi.all_states()), axis=0)
    run = jax.jit(
        lambda p, x, H: kernels.local_value_kernel_jax_unique(
            logpsi, p, x, H, chunk_size=3
        )
    )
    executable = run.lower(0.17, x, H).compile()
    for name in ("all-gather", "all-reduce", "all-to-all", "collective-permute"):
        assert name not in executable.as_text()


@pytest.mark.parametrize("empty", [True, False])
def test_empty_and_diagonal_operators(empty):
    hi = nk.hilbert.Spin(0.5, 4)
    H = nk.operator.LocalOperatorJax(hi)
    if not empty:
        H += nk.operator.spin.sigmaz(hi, 0)
    x = jnp.repeat(jnp.asarray(hi.all_states())[:2], 8, axis=0)
    x = nk.jax.sharding.shard_along_axis(x, axis=0)
    observed = []

    def counted(p, rows):
        jax.debug.callback(
            lambda rows: observed.extend(map(tuple, np.asarray(rows))), rows
        )
        return logpsi(p, rows)

    run = jax.jit(
        lambda p, x, H: kernels.local_value_kernel_jax_unique(
            counted, p, x, H, chunk_size=3
        )
    )
    result = jax.block_until_ready(run(0.2, x, H))
    jax.effects_barrier()
    if empty:
        np.testing.assert_array_equal(result, np.zeros(len(x)))
        assert observed == []
    else:
        expected = kernels.local_value_kernel_jax(logpsi, 0.2, x, H)
        np.testing.assert_array_equal(result, expected)
        shards = jax.device_count() if nk.config.netket_sharding else 1
        expected_rows = [
            tuple(row)
            for local in np.split(np.asarray(x), shards)
            for row in np.unique(local, axis=0)
        ]
        assert Counter(observed) == Counter(expected_rows)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_repeated_expansion_transpose_is_bitwise_stable(dtype):
    # Deliberately large groups exercise the floating-point accumulation in AD.
    # Use an analytic model so this isolates the deduplication algorithm from
    # any nondeterminism in a neural network's own backward pass.
    rows = (jnp.arange(4096, dtype=jnp.int32) % 7)[:, None]
    order, starts, positions, _, _ = kernels._unique_row_plan(rows)
    values = jnp.cos(jnp.arange(4096, dtype=jnp.float64)).astype(dtype)
    ct = jnp.sin(jnp.arange(4096, dtype=jnp.float64)).astype(dtype)
    if dtype == jnp.complex128:
        values = values * (1 + 0.3j)
        ct = ct * (1 - 0.7j)
    run = jax.jit(
        lambda v, ct: jax.vjp(
            lambda v: kernels._expand_unique_rows(v, order, starts, positions), v
        )[1](ct)[0]
    )
    results = [np.asarray(run(values, ct)) for _ in range(10)]
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    expected = np.zeros_like(results[0])
    for group, value in zip(np.asarray(rows[:, 0]), np.asarray(ct)):
        expected[group] += value
    np.testing.assert_allclose(results[0], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_mcstate_dispatch_and_expect_grad(chunk_size):
    hi = nk.hilbert.Spin(0.5, 4)
    H = RepeatedConnections(nk.operator.IsingJax(hi, nk.graph.Chain(4), h=0.7))
    vs = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains=8),
        nk.models.RBM(alpha=1, param_dtype=jnp.complex128),
        n_samples=32,
        seed=17,
    )
    old = nk.config.netket_experimental_unique_kernel
    experimental = nk.config.netket_experimental
    flattened = nk.config.netket_experimental_flattened_kernel
    try:
        nk.config.netket_experimental = True
        nk.config.netket_experimental_unique_kernel = False
        nk.config.netket_experimental_flattened_kernel = False
        expected = np.asarray(vs.local_estimators(H, chunk_size=chunk_size).data)
        e0, g0 = vs.expect_and_grad(H, use_covariance=False)
        nk.config.netket_experimental_unique_kernel = True
        assert (
            nk.vqs.get_local_kernel(vs, H, chunk_size)
            is kernels.local_value_kernel_jax_fingerprint
        )
        actual = np.asarray(vs.local_estimators(H, chunk_size=chunk_size).data)
        e1, g1 = vs.expect_and_grad(H, use_covariance=False)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(e1.mean, e0.mean, rtol=1e-12, atol=1e-12)
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12),
            g1,
            g0,
        )
    finally:
        nk.config.netket_experimental_unique_kernel = old
        nk.config.netket_experimental = experimental
        nk.config.netket_experimental_flattened_kernel = flattened
