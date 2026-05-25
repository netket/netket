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

"""Tests for LocalEstimators and LocalEstimatorsBatch."""

import warnings

import numpy as np
import numpy.testing as npt
import pytest
import jax.numpy as jnp

import netket as nk
from netket.stats import LocalEstimators, LocalEstimatorsBatch, statistics
from netket._src.stats.online_stats import OnlineStats
from netket._src.stats.online_stats.accumulator_batch import OnlineStatsBatch

from .. import common


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def scalar_data():
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.standard_normal((8, 50)))  # (n_chains, chain_len)


@pytest.fixture
def batch_data():
    rng = np.random.default_rng(1)
    return jnp.asarray(rng.standard_normal((8, 50, 2)))  # (n_chains, chain_len, K=2)


def variance_combinator(mu):
    return mu[1] - mu[0] ** 2


def matrix_combinator(mu):
    return jnp.outer(mu, mu)  # (K,) -> (K, K)


# ── LocalEstimators ───────────────────────────────────────────────────────────


def test_to_stats_matches_statistics(scalar_data):
    le = LocalEstimators(scalar_data)
    stats = le.to_stats()
    ref = statistics(scalar_data)
    npt.assert_allclose(float(stats.mean), float(ref.mean), rtol=1e-6)
    npt.assert_allclose(
        float(stats.error_of_mean), float(ref.error_of_mean), rtol=1e-6
    )


def test_accumulate(scalar_data):
    le = LocalEstimators(scalar_data)

    # fresh accumulation returns OnlineStats with correct sample count
    acc = le.accumulate()
    assert isinstance(acc, OnlineStats)
    assert acc.n_samples == scalar_data.size

    # updating increments sample count
    acc2 = le.accumulate(acc)
    assert acc2.n_samples == 2 * scalar_data.size

    # mean is consistent with one-shot to_stats()
    npt.assert_allclose(acc2.mean, float(le.to_stats().mean), rtol=1e-6)


def test_array_interface(scalar_data):
    le = LocalEstimators(scalar_data)
    assert le.shape == scalar_data.shape
    assert le.ndim == scalar_data.ndim
    assert le.size == scalar_data.size
    assert le.dtype == scalar_data.dtype
    npt.assert_allclose(jnp.asarray(le), scalar_data)


def test_unknown_attr_raises(scalar_data):
    le = LocalEstimators(scalar_data)
    with pytest.raises(AttributeError):
        _ = le.nonexistent_attr


# ── LocalEstimatorsBatch ──────────────────────────────────────────────────────


def test_batch_to_stats_scalar_combinator(batch_data):
    le = LocalEstimatorsBatch(batch_data, variance_combinator)
    stats = le.to_stats()
    assert isinstance(stats, nk.stats.Stats)
    assert np.isfinite(float(stats.mean))
    assert float(stats.error_of_mean) > 0


def test_batch_to_stats_array_combinator(batch_data):
    le = LocalEstimatorsBatch(batch_data, matrix_combinator)
    stats = le.to_stats()
    # Array-valued combinator → StatsBatch with .mean and .error_of_mean
    assert stats.mean.shape == (2, 2)
    assert stats.error_of_mean.shape == (2, 2)
    assert np.all(np.isfinite(stats.mean))


def test_batch_to_stats_scalar_value_is_sensible():
    """variance_combinator(mu) = E[X²] - E[X]² matches scalar statistics."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((8, 50))
    # channel 0: x, channel 1: x²  → combinator gives Var[X]
    data = jnp.asarray(np.stack([x, x**2], axis=-1))

    le_batch = LocalEstimatorsBatch(data, variance_combinator)
    var_batch = float(le_batch.to_stats().mean)

    # Reference: per-chain means then grand mean for each channel
    mu_x = float(np.mean(np.mean(x, axis=1)))
    mu_x2 = float(np.mean(np.mean(x**2, axis=1)))
    var_manual = mu_x2 - mu_x**2

    npt.assert_allclose(var_batch, var_manual, rtol=1e-5)


def test_batch_accumulate(batch_data):
    le = LocalEstimatorsBatch(batch_data, variance_combinator)

    # fresh accumulation returns OnlineStatsBatch with a valid Stats
    acc = le.accumulate()
    assert isinstance(acc, OnlineStatsBatch)
    assert np.isfinite(float(acc.get_stats().mean))

    # updating increments sample count
    acc2 = le.accumulate(acc)
    assert acc2.n_samples > acc.n_samples

    # mean is consistent with one-shot to_stats()
    npt.assert_allclose(float(acc2.get_stats().mean), float(le.to_stats().mean), rtol=1e-5)
