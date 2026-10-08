# Copyright 2026 The NetKet Authors - All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Safe fingerprint collisions and device-local heuristic branch selection."""
from collections import Counter

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import pytest
from netket.vqs.mc import kernels


def logpsi(p, x):
    return p * jnp.sum(x * jnp.arange(1, x.shape[-1] + 1), axis=-1) + 0.03j * x[..., 0]


@pytest.mark.parametrize(
    "dtype", [jnp.int8, jnp.int64, jnp.float32, jnp.float64, jnp.complex128]
)
@pytest.mark.parametrize("collisions", [False, True])
def test_fingerprint_groups_never_merge_unequal_rows(dtype, collisions):
    rows = jnp.array(
        [[1, 2], [3, 4], [1, 2], [1, 2], [5, 6], [3, 4], [3, 4]], dtype=dtype
    )
    keys = jnp.zeros(len(rows), dtype=jnp.uint32) if collisions else None
    order, starts, positions, representatives, count = kernels._fingerprint_row_plan(
        rows, keys
    )
    f = jax.jit(
        lambda p: kernels._expand_unique_rows(
            logpsi(p, rows[representatives]), order, starts, positions
        )
    )
    np.testing.assert_allclose(f(0.17), logpsi(0.17, rows), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        jax.jacfwd(f)(0.17),
        jax.jacfwd(lambda p: logpsi(p, rows))(0.17),
        rtol=1e-12,
        atol=1e-12,
    )
    if collisions:
        assert int(count) > len(np.unique(np.asarray(rows), axis=0))


def test_runtime_gate_switches_without_retracing_or_changing_results():
    hi = nk.hilbert.Spin(0.5, 4)
    H = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=0.7)
    states = jnp.asarray(hi.all_states())
    observed, traces = [], []

    def counted(p, x):
        traces.append(x.shape)
        jax.debug.callback(lambda x: observed.extend(map(tuple, np.asarray(x))), x)
        return logpsi(p, x)

    run = jax.jit(
        lambda p, x, H: kernels.local_value_kernel_jax_reuse(
            counted, p, x, H, chunk_size=3
        )
    )
    for k, x in enumerate(
        (
            jnp.repeat(states[:1], 32, axis=0),
            jnp.tile(states, (2, 1)),
            jnp.concatenate((jnp.repeat(states[:1], 16, axis=0), states)),
        )
    ):
        x = nk.jax.sharding.shard_along_axis(x, axis=0)
        observed.clear()
        actual = jax.block_until_ready(run(0.17, x, H))
        jax.effects_barrier()
        expected = kernels.local_value_kernel_jax(logpsi, 0.17, x, H)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        if k == 0:
            n_traces = len(traces)
        assert len(traces) == n_traces
        n_devices = jax.device_count() if nk.config.netket_sharding else 1
        expected_rows = []
        for local in np.split(np.asarray(x), n_devices):
            xp, _ = H.get_conn_padded(jnp.asarray(local))
            xp = np.asarray(xp)
            if bool(kernels._sample_reuse_predicate(jnp.asarray(local))):
                rows = np.concatenate((local, xp.reshape(-1, hi.size)))
                # This finite fixture has no fingerprint collisions.
                expected_rows.extend(map(tuple, np.unique(rows, axis=0)))
            else:
                expected_rows.extend(map(tuple, local))
                expected_rows.extend(
                    map(tuple, xp[np.any(xp != local[:, None], axis=-1)])
                )
        assert Counter(observed) == Counter(expected_rows)


@pytest.mark.parametrize("repeated", [False, True])
def test_both_branches_support_operator_model_and_second_derivatives(repeated):
    hi = nk.hilbert.Spin(0.5, 4)
    states = jnp.asarray(hi.all_states())
    x = jnp.repeat(states[:1], 16, axis=0) if repeated else states
    x = nk.jax.sharding.shard_along_axis(x, axis=0)

    def loss(p, kernel):
        H = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=p[1])
        return jnp.sum(jnp.abs(kernel(logpsi, p[0], x, H)) ** 2)

    def reuse(*args):
        return kernels.local_value_kernel_jax_reuse(*args, chunk_size=3)

    p = jnp.array([0.17, 0.0])
    for transform in (jax.grad, lambda f: jax.jacfwd(jax.jacrev(f))):
        actual = jax.jit(transform(lambda p: loss(p, reuse)))(p)
        expected = jax.jit(
            transform(lambda p: loss(p, kernels.local_value_kernel_jax))
        )(p)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def test_forced_fingerprint_collisions_preserve_local_energies(monkeypatch):
    monkeypatch.setattr(
        kernels,
        "_row_fingerprints",
        lambda rows: jnp.zeros(rows.shape[0], dtype=jnp.uint32),
    )
    hi = nk.hilbert.Spin(0.5, 4)
    H = nk.operator.IsingJax(hi, nk.graph.Chain(4), h=0.7)
    x = nk.jax.sharding.shard_along_axis(jnp.asarray(hi.all_states()), axis=0)
    run = jax.jit(
        lambda p, x, H: kernels.local_value_kernel_jax_reuse(
            logpsi, p, x, H, chunk_size=3, check_samples=False
        )
    )
    actual = run(0.17, x, H)
    expected = kernels.local_value_kernel_jax(logpsi, 0.17, x, H)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
