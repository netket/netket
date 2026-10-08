# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Bounded coarse-tail work, discarded filler derivatives, and API propagation."""

from collections import Counter

import jax
import jax.numpy as jnp
import netket as nk
from netket.vqs.mc import kernels
import numpy as np
import pytest


@pytest.mark.parametrize(
    "chunk,requested,minimum",
    [
        (1, None, 1),
        (7, 3, 2),
        (7, 4, 4),
        (10, 8, 8),
        (7, 7, 7),
        (16, None, 2),
        (32, None, 4),
        (128, None, 16),
        (7, 100, 7),
        (7, 0, 1),
    ],
)
@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_coarse_tail_work_and_filler_derivatives(chunk, requested, minimum, mode):
    assert kernels._flattened_min_chunk_size(chunk, requested) == minimum
    capacity = 3 * chunk
    buffer_size = capacity + minimum - 1
    rows = jnp.arange(capacity + 1, dtype=float).reshape(-1, 1)
    observed, traces = [], []

    def record(x):
        assert minimum <= len(x) <= chunk
        observed.extend(np.asarray(x)[:, 0])

    def counted(p, x):
        traces.append(x.shape)
        # eval_shape also traces a single row to infer the output dtype. Count
        # and validate only batches that actually execute on the device.
        jax.debug.callback(record, x)
        return p * x[:, 0]

    def values(p, count):
        positions = jnp.arange(buffer_size)
        indices = jnp.where(positions < count, positions, capacity)
        result = kernels._flattened_logpsi(
            counted, chunk, p, rows, indices, count, min_chunk_size=minimum
        )
        # The public kernels consume only occupied values. Filler outputs and
        # their parameter derivatives must not influence that result.
        return jnp.where(positions < count, result, 0)

    if mode == "jvp":
        run = jax.jit(lambda p, n: jax.jvp(lambda p: values(p, n), (p,), (1.0,)))
    elif mode == "vjp":
        run = jax.jit(jax.value_and_grad(lambda p, n: jnp.sum(values(p, n))))
    else:
        run = jax.jit(values)

    counts = (
        range(capacity + 1)
        if chunk <= 16
        else [
            0,
            1,
            minimum - 1,
            minimum,
            minimum + 1,
            chunk - 1,
            chunk,
            chunk + 1,
            2 * chunk - 1,
            capacity - minimum,
            capacity - 1,
            capacity,
        ]
    )
    for iteration, count in enumerate(counts):
        observed.clear()
        actual = jax.block_until_ready(run(0.25, jnp.asarray(count)))
        jax.effects_barrier()
        remainder = count % chunk
        extra = (-remainder) % minimum
        assert 0 <= extra < minimum
        assert Counter(observed) == Counter(list(range(count)) + [capacity] * extra)
        expected = np.zeros(buffer_size)
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


@pytest.mark.parametrize(
    "kernel",
    [
        kernels.local_value_kernel_jax_flattened,
        kernels.local_value_kernel_jax_unique,
        kernels.local_value_kernel_jax_fingerprint,
        kernels.local_value_kernel_jax_reuse,
    ],
)
@pytest.mark.parametrize("minimum", [1, None, 7, 128])
def test_public_tail_options_preserve_values_and_zero_coefficient_ad(kernel, minimum):
    hi = nk.hilbert.Spin(0.5, 4)
    graph = nk.graph.Chain(4)
    # Repeated configurations exercise reuse and leave partially occupied tails.
    x = jnp.repeat(jnp.asarray(hi.all_states())[:1], 16, axis=0)
    x = nk.jax.sharding.shard_along_axis(x, axis=0)

    def logpsi(p, rows):
        return p * jnp.sum(rows, axis=-1) + 0.03j * rows[..., 0]

    def loss(p, compact):
        operator = nk.operator.IsingJax(hi, graph, h=p[1], J=0.7)
        if compact:
            values = kernel(
                logpsi, p[0], x, operator, chunk_size=128, min_chunk_size=minimum
            )
        else:
            values = kernels.local_value_kernel_jax(logpsi, p[0], x, operator)
        return jnp.sum(jnp.abs(values) ** 2)

    p = jnp.array([0.17, 0.0])
    expected = jax.jit(jax.value_and_grad(lambda p: loss(p, False)))(p)
    actual = jax.jit(jax.value_and_grad(lambda p: loss(p, True)))(p)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-10),
        actual,
        expected,
    )
