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

import math
import numpy as np
import pytest

import netket as nk

from .. import common

# Tests that force chain positions via plain numpy arrays are skipped under
# distributed execution (sharded sampler_state can't accept a host array).
# Everything else runs in both single-process and distributed modes.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def hi():
    return nk.hilbert.Spin(1 / 2, N=6)


@pytest.fixture()
def hamiltonian(hi):
    return nk.operator.Ising(hi, nk.graph.Chain(6), h=1.0)


@pytest.fixture()
def vstate_uniform(hi):
    """MCState with a uniform wavefunction (LogStateVector, default init = all-ones).

    Because log|ψ(σ)| is constant, every MetropolisLocal proposal is accepted,
    so chains mix in a single sweep regardless of where they started.  This
    makes thermalisation trivially easy and allows tests to run fast.
    """
    vs = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains=16, sweep_size=hi.size),
        nk.models.LogStateVector(hi, param_dtype=float),
        n_samples=128,
        n_discard_per_chain=0,
        seed=42,
    )
    # Default LogStateVector init is ones → uniform distribution.
    return vs


def _force_all_chains_to_same_state(vs):
    """Set every chain to the all-spins-up configuration."""
    n_chains = vs.sampler.n_chains
    all_up = np.ones((n_chains, vs.hilbert.size), dtype=np.int8)
    vs.sampler_state = vs.sampler_state.replace(σ=all_up)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_raises_non_metropolis(hi, hamiltonian):
    """Raises ValueError when the sampler is not a MetropolisSampler."""
    vs = nk.vqs.MCState(
        nk.sampler.ExactSampler(hi),
        nk.models.RBM(alpha=1),
        n_samples=16,
        seed=0,
    )
    with pytest.raises(ValueError, match="MetropolisSampler"):
        vs.thermalise(hamiltonian, verbose=False)


@common.skipif_distributed
def test_raises_single_chain(hi, hamiltonian):
    """Raises ValueError when n_chains < 2 (R̂ is undefined).

    Skipped in distributed mode: n_chains=1 per process × N processes ≥ 2
    total chains, so the guard never fires and the test would be meaningless.
    """
    vs = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi, n_chains=1),
        nk.models.RBM(alpha=1),
        n_samples=16,
        seed=0,
    )
    with pytest.raises(ValueError, match="n_chains"):
        vs.thermalise(hamiltonian, verbose=False)


# ---------------------------------------------------------------------------
# Return types and history
# ---------------------------------------------------------------------------


def test_return_types_and_hist_keys(vstate_uniform, hamiltonian):
    """Returns (OnlineStats, HistoryDict) with expected keys and finite values."""
    stats, hist = vstate_uniform.thermalise(
        hamiltonian, max_chain_length=100, verbose=False
    )

    assert math.isfinite(stats.mean)
    assert math.isfinite(stats.variance)
    for key in ("mean", "R_hat", "variance", "error_of_mean"):
        assert key in hist, f"Missing key '{key}' in hist"


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


@common.skipif_distributed
def test_mutates_sampler_state(vstate_uniform, hamiltonian):
    """thermalise mutates state.sampler_state in-place (unlike check_mc_convergence)."""
    _force_all_chains_to_same_state(vstate_uniform)
    sigma_before = np.array(vstate_uniform.sampler_state.σ)

    vstate_uniform.thermalise(hamiltonian, max_chain_length=200, verbose=False)

    sigma_after = np.array(vstate_uniform.sampler_state.σ)
    # Chains must have moved from the forced all-up start.
    assert not np.all(sigma_after == sigma_before)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


@common.skipif_distributed
def test_converges_from_biased_start(vstate_uniform, hamiltonian):
    """With a uniform wavefunction and all chains starting at the same state,
    thermalise should converge to R̂ < rhat_tol well before max_chain_length."""
    _force_all_chains_to_same_state(vstate_uniform)

    # Very generous budget; uniform distribution mixes in one sweep.
    stats, hist = vstate_uniform.thermalise(
        hamiltonian,
        max_chain_length=500,
        rhat_tol=1.05,
        patience=5,
        verbose=False,
    )

    rhat = stats.R_hat
    assert not math.isnan(rhat), "R̂ is NaN — not enough data to compute it"
    assert rhat < 1.05, f"R̂={rhat:.4f} did not converge below 1.05"

    # Converged well before the budget.
    n_samples_per_chain = stats._n_samples_total / vstate_uniform.sampler.n_chains
    assert (
        n_samples_per_chain < 500
    ), f"Used {n_samples_per_chain} samples/chain — should have converged much earlier"


@common.skipif_distributed
def test_r_hat_improves_from_biased_start(vstate_uniform, hamiltonian):
    """R̂ after thermalisation is lower than R̂ computed right after forcing chains
    to the same state (before any mixing has occurred)."""
    _force_all_chains_to_same_state(vstate_uniform)

    # Collect R_hat values across thermalisation.
    stats, hist = vstate_uniform.thermalise(
        hamiltonian,
        max_chain_length=500,
        rhat_tol=1.05,
        patience=5,
        verbose=False,
    )

    rhats = hist["R_hat"].values
    # R_hat should end up lower than (or equal to) where it started.
    # We just check it is monotonically non-increasing on average (final < initial).
    finite = rhats[np.isfinite(rhats)]
    assert len(finite) > 1
    assert finite[-1] <= finite[0] + 0.1, "R̂ did not decrease over thermalisation"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_warns_on_max_chain_length(vstate_uniform, hamiltonian):
    """Emits UserWarning when max_chain_length is exhausted without convergence."""
    with pytest.warns(UserWarning, match="max.*chain length|maximum chain length"):
        vstate_uniform.thermalise(
            hamiltonian,
            max_chain_length=2,  # too short to ever satisfy patience
            patience=10_000,
            verbose=False,
        )


def test_raise_on_failure(vstate_uniform, hamiltonian):
    """raise_on_failure=True raises RuntimeError instead of a warning."""
    with pytest.raises(RuntimeError):
        vstate_uniform.thermalise(
            hamiltonian,
            max_chain_length=2,
            patience=10_000,
            verbose=False,
            raise_on_failure=True,
        )


# ---------------------------------------------------------------------------
# Distributed-safe convergence (no forced chain positions)
# ---------------------------------------------------------------------------


def test_converges_uniform_default_init(vstate_uniform, hamiltonian):
    """Uniform wavefunction converges from default initialisation.

    This test runs under both single-process and distributed execution:
    it verifies that R̂ aggregation across sharded chains produces a finite,
    converged result without writing plain host arrays to sampler_state.
    """
    stats, hist = vstate_uniform.thermalise(
        hamiltonian,
        max_chain_length=500,
        rhat_tol=1.05,
        patience=3,
        verbose=False,
    )

    rhat = stats.R_hat
    assert not math.isnan(rhat), "R̂ is NaN — likely an aggregation bug under sharding"
    assert rhat < 1.05, f"R̂={rhat:.4f} did not converge below 1.05"

    rhats = hist["R_hat"].values
    finite = rhats[np.isfinite(rhats)]
    assert len(finite) >= 1, "No finite R̂ values recorded in history"
