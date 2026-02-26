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
from netket.stats import online_statistics
from netket._src.vqs.check_mc_convergence import acf_window_saturated, tau_corr_reliable

from .. import common

pytestmark = common.skipif_distributed


def ar1(phi, n_chains=8, n_samples=500, seed=42):
    """AR(1) process x[t] = phi*x[t-1] + sqrt(1-phi^2)*noise, tau ~ (1+phi)/(1-phi)."""
    rng = np.random.default_rng(seed)
    data = np.zeros((n_chains, n_samples))
    data[:, 0] = rng.standard_normal(n_chains)
    for t in range(1, n_samples):
        data[:, t] = phi * data[:, t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(
            n_chains
        )
    return data


@pytest.fixture()
def vstate():
    hi = nk.hilbert.Spin(1 / 2, N=4)
    return nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        nk.models.RBM(alpha=1),
        n_samples=16,
        seed=0,
    )


# --- acf_window_saturated ---


def test_acf_saturated_small_window():
    """Strongly correlated AR(1) saturates a small ACF window."""
    est = online_statistics(ar1(phi=0.9), max_lag=8)
    assert acf_window_saturated(est)


def test_acf_not_saturated_large_window():
    """AR(1) phi=0.9 does NOT saturate a large window (tau ~ 19, max_lag=64 >> tau)."""
    est = online_statistics(ar1(phi=0.9), max_lag=64)
    assert not acf_window_saturated(est)


def test_acf_not_saturated_iid():
    """iid data is never saturated."""
    est = online_statistics(ar1(phi=0.0), max_lag=32)
    assert not acf_window_saturated(est)


# --- tau_corr_reliable ---


def test_tau_reliable_saturated_is_false():
    """tau is not reliable when the window is saturated."""
    est = online_statistics(ar1(phi=0.9), max_lag=8)
    assert not tau_corr_reliable(est)


def test_tau_reliable_iid_enough_data():
    """iid data with plenty of samples yields a reliable tau."""
    est = online_statistics(ar1(phi=0.0, n_samples=500), max_lag=32)
    assert tau_corr_reliable(est)


def test_tau_reliable_too_few_samples():
    """Too few samples → n_eff/tau < 50 → not reliable."""
    est = online_statistics(ar1(phi=0.0, n_samples=5), max_lag=4)
    assert not tau_corr_reliable(est)


# --- check_mc_convergence integration ---


def test_check_mc_convergence_runs(vstate):
    """check_mc_convergence returns (OnlineStats, HistoryDict) with finite stats."""
    H = nk.operator.Ising(vstate.hilbert, nk.graph.Chain(4), h=1.0)
    stats, hist = vstate.check_mc_convergence(H, max_chain_length=50)

    assert math.isfinite(stats.mean)
    assert math.isfinite(stats.variance)
    assert "mean" in hist and "tau_corr_acf" in hist


def test_check_mc_convergence_plot(vstate):
    """plot=True runs without error and does not open a window."""
    from unittest.mock import patch

    H = nk.operator.Ising(vstate.hilbert, nk.graph.Chain(4), h=1.0)
    with patch("matplotlib.pyplot.show"):
        vstate.check_mc_convergence(H, max_chain_length=50, plot=True)
