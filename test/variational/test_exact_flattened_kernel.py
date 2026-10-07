# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Exact network work, including partial chunks and differentiated execution."""

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import netket as nk
from netket.vqs.mc import kernels


def logpsi(p, x):
    return p * jnp.sum(x, axis=-1) + 0.03j * jnp.sum(x[..., ::2], axis=-1)


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16])
@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_every_remainder_evaluates_exactly_valid_rows(chunk_size, mode):
    # Exercise every count, not just one convenient remainder. The sentinel
    # index belongs to allocated storage and must never reach the network.
    capacity = 3 * chunk_size
    rows = jnp.arange(capacity + 1, dtype=float).reshape(-1, 1)
    observed = []
    traces = []

    def counted(p, rows):
        traces.append(rows.shape)
        assert 0 < rows.shape[0] <= chunk_size
        jax.debug.callback(lambda x: observed.extend(np.asarray(x)[:, 0]), rows)
        return p * rows[:, 0]

    def values(p, count):
        idx = jnp.where(jnp.arange(capacity) < count, jnp.arange(capacity), capacity)
        return kernels._flattened_logpsi(counted, chunk_size, p, rows, idx, count)

    if mode == "jvp":
        run = jax.jit(
            lambda p, count: jax.jvp(lambda p: values(p, count), (p,), (1.0,))
        )
    elif mode == "vjp":
        run = jax.jit(jax.value_and_grad(lambda p, count: jnp.sum(values(p, count))))
    else:
        run = jax.jit(values)

    for count in range(capacity + 1):
        observed.clear()
        result = jax.block_until_ready(run(0.25, jnp.asarray(count)))
        jax.effects_barrier()
        assert sorted(observed) == list(range(count))
        expected = np.zeros(capacity)
        expected[:count] = 0.25 * np.arange(count)
        if mode == "jvp":
            np.testing.assert_allclose(result[0], expected)
            np.testing.assert_allclose(result[1], expected * 4)
        elif mode == "vjp":
            np.testing.assert_allclose(result[0], expected.sum())
            np.testing.assert_allclose(result[1], expected.sum() * 4)
        else:
            np.testing.assert_allclose(result, expected)
        if count == 0:
            n_traces = len(traces)
        assert len(traces) == n_traces


@pytest.mark.parametrize("full_capacity", [7, 10, 17])
@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_full_chunk_blocks_cross_boundaries_without_extra_rows_or_retracing(
    full_capacity, mode
):
    # Cover capacities both on and between binary boundaries, including AD.
    chunk_size, capacity = 5, full_capacity * 5
    rows = jnp.arange(capacity + 1, dtype=float).reshape(-1, 1)
    observed, traces = [], []

    def counted(p, batch):
        traces.append(batch.shape)
        assert 0 < len(batch) <= chunk_size
        jax.debug.callback(lambda x: observed.extend(np.asarray(x)[:, 0]), batch)
        return p * batch[:, 0]

    def values(p, count):
        indices = jnp.where(
            jnp.arange(capacity) < count, jnp.arange(capacity), capacity
        )
        return kernels._flattened_logpsi(counted, chunk_size, p, rows, indices, count)

    if mode == "jvp":
        run = jax.jit(lambda p, n: jax.jvp(lambda p: values(p, n), (p,), (1.0,)))
    elif mode == "vjp":
        run = jax.jit(jax.value_and_grad(lambda p, n: jnp.sum(values(p, n))))
    else:
        run = jax.jit(values)
    counts = (capacity, 0, 41, 1, 4, 5, 9, 10, 20, 21, 39, 40, 64, 79, 80, 81, 84)
    for iteration, count in enumerate(n for n in counts if n <= capacity):
        observed.clear()
        actual = jax.block_until_ready(run(0.25, jnp.asarray(count)))
        jax.effects_barrier()
        assert sorted(observed) == list(range(count))
        expected = np.zeros(capacity)
        expected[:count] = 0.25 * np.arange(count)
        if mode == "jvp":
            np.testing.assert_array_equal(actual[0], expected)
            np.testing.assert_array_equal(actual[1], expected * 4)
        elif mode == "vjp":
            np.testing.assert_array_equal(actual[0], expected.sum())
            np.testing.assert_array_equal(actual[1], expected.sum() * 4)
        else:
            np.testing.assert_array_equal(actual, expected)
        if iteration == 0:
            n_traces = len(traces)
        assert len(traces) == n_traces


@jax.tree_util.register_pytree_node_class
class VariableConnections(nk.operator.DiscreteJaxOperator):
    """An existing-interface operator with uneven, sample-filled padding."""

    @property
    def dtype(self):
        return np.dtype(float)

    @property
    def max_conn_size(self):
        return self.hilbert.size + 2

    def tree_flatten(self):
        return (), self.hilbert

    @classmethod
    def tree_unflatten(cls, hilbert, children):
        return cls(hilbert)

    def get_conn_padded(self, x):
        sites = jnp.eye(x.shape[-1], dtype=bool)
        active = x < 0
        flips = jnp.where(sites[None] & active[:, :, None], -x[:, None], x[:, None])
        # Diagonal terms at both ends check that no fixed position is assumed.
        xp = jnp.concatenate((x[:, None], flips, x[:, None]), axis=1)
        mels = jnp.concatenate(
            (x[:, :1] * 0 + 0.3, active * 0.7, x[:, :1] * 0 + 0.2), axis=1
        )
        return xp, mels


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16])
def test_existing_operator_interface_and_exact_work(chunk_size):
    hi = nk.hilbert.Spin(0.5, 4)
    H = VariableConnections(hi)
    x = nk.jax.sharding.shard_along_axis(jnp.asarray(hi.all_states()), axis=0)
    observed = []

    def counted(p, rows):
        assert rows.shape[0] <= chunk_size
        jax.debug.callback(
            lambda rows: observed.extend(map(tuple, np.asarray(rows))), rows
        )
        return logpsi(p, rows)

    run = jax.jit(
        lambda p, x, H: kernels.local_value_kernel_jax_flattened(
            counted, p, x, H, chunk_size=chunk_size
        )
    )
    actual = jax.block_until_ready(run(0.2, x, H))
    jax.effects_barrier()
    expected = kernels.local_value_kernel_jax(logpsi, 0.2, x, H)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    xp, mels = H.get_conn_padded(x)
    is_conn = np.any(np.asarray(xp) != np.asarray(x)[:, None], axis=-1)
    expected_rows = list(map(tuple, np.asarray(x))) + list(
        map(tuple, np.asarray(xp)[is_conn])
    )
    assert Counter(observed) == Counter(expected_rows)


@pytest.mark.parametrize("chunk_size", [3, 16])
def test_forward_reverse_and_mixed_second_derivatives(chunk_size):
    hi = nk.hilbert.Spin(0.5, 4)
    graph = nk.graph.Chain(4)
    x = nk.jax.sharding.shard_along_axis(jnp.asarray(hi.all_states()), axis=0)

    def loss(p, kernel):
        # A zero physical coefficient is not padding: cross derivatives need
        # its connected wavefunction even at h=0.
        H = nk.operator.IsingJax(hi, graph, h=p[1], J=0.7)
        values = kernel(logpsi, p[0], x, H)
        return jnp.sum(jnp.abs(values) ** 2)

    def exact(*args):
        return kernels.local_value_kernel_jax_flattened(*args, chunk_size=chunk_size)

    p, tangent = jnp.array([0.17, 0.0]), jnp.array([0.3, 0.8])
    for transform in (
        lambda f: jax.jit(lambda p: jax.jvp(f, (p,), (tangent,))),
        lambda f: jax.jit(jax.grad(f)),
        lambda f: jax.jit(jax.jacfwd(jax.jacrev(f))),
    ):
        expected = transform(lambda p: loss(p, kernels.local_value_kernel_jax))(p)
        actual = transform(lambda p: loss(p, exact))(p)
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11),
            actual,
            expected,
        )
