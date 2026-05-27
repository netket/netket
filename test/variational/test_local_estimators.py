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

"""Tests for MCState.local_estimators integration."""

import numpy as np
import numpy.testing as npt
import pytest
import jax

import netket as nk
from netket.stats import LocalEstimators

from .. import common


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hi():
    return nk.hilbert.Spin(0.5, 4)


@pytest.fixture(scope="module")
def vstate(hi):
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
    ma = nk.models.RBM(alpha=1)
    return nk.vqs.MCState(sa, ma, n_samples=256, seed=0)


@pytest.fixture
def vstate_fn(hi):
    sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
    ma = nk.models.RBM(alpha=1)
    return nk.vqs.MCState(sa, ma, n_samples=256, seed=0)


@pytest.fixture(scope="module")
def ising(hi):
    g = nk.graph.Chain(4, pbc=True)
    return nk.operator.Ising(hi, g, h=1.0)


# ── MCState.local_estimators (non-sharding) ───────────────────────────────────


@common.skipif_distributed
def test_local_estimators(vstate, ising):
    le = vstate.local_estimators(ising)

    # type and shape
    assert isinstance(le, LocalEstimators)
    assert le.data.shape == (vstate.sampler.n_chains, vstate.chain_length)

    # to_stats() agrees with expect()
    stats_le = le.to_stats()
    stats_ex = vstate.expect(ising)
    npt.assert_allclose(float(stats_le.mean.real), float(stats_ex.mean.real), rtol=1e-6)
    npt.assert_allclose(
        float(stats_le.error_of_mean), float(stats_ex.error_of_mean), rtol=1e-6
    )

    # chunked evaluation gives identical data
    le_chunked = vstate.local_estimators(ising, chunk_size=16)
    npt.assert_allclose(np.array(le.data), np.array(le_chunked.data), atol=1e-5)


@common.skipif_distributed
def test_accumulation_across_steps(vstate_fn, ising):
    vstate_fn.sample()
    acc = vstate_fn.local_estimators(ising).accumulate()
    vstate_fn.sample(n_discard_per_chain=0)
    acc = vstate_fn.local_estimators(ising).accumulate(acc)

    stats = acc.get_stats()
    assert np.isfinite(float(stats.mean.real))
    assert float(stats.error_of_mean) > 0


@common.onlyif_distributed
def test_local_estimators_sharding_matches_samples(vstate_fn, ising):
    samples = vstate_fn.samples
    le = vstate_fn.local_estimators(ising)
    assert le.data.sharding.is_equivalent_to(samples.sharding, le.data.ndim)


# ── sharding ──────────────────────────────────────────────────────────────────


@common.onlyif_sharding_single_process
def test_local_estimators_under_sharding(vstate, ising):
    le = vstate.local_estimators(ising)
    jax.block_until_ready(le.data)
    le.to_stats()
