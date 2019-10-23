import numpy as np
import pytest

import netket as nk
import netket.variational as vmc
from netket.operator import local_values
from netket.stats import statistics


def _setup():
    g = nk.graph.Hypercube(3, 2)
    hi = nk.hilbert.Spin(g, 0.5)

    ham = nk.operator.Heisenberg(hi)

    ma = nk.machine.RbmSpin(hi, alpha=2)
    ma.init_random_parameters()

    return hi, ham, ma


def _test_stats_mean_std(hi, ham, ma, n_chains):
    sampler = nk.sampler.MetropolisLocal(ma, n_chains=n_chains)

    n_samples = 16000
    num_samples_per_chain = n_samples // n_chains

    samples = nk.sampler.compute_samples(sampler, n_samples=n_samples, n_discard=6400)
    assert samples.shape == (num_samples_per_chain, n_chains, hi.size)

    eloc = local_values(ham, ma, samples)
    assert eloc.shape == (num_samples_per_chain, n_chains)

    stats = statistics(eloc)

    # These tests only work for one MPI process
    assert nk.MPI.size() == 1

    assert stats.mean == pytest.approx(np.mean(eloc))
    if n_chains > 1:
        # error of mean == stdev of sample mean between chains / sqrt(#chains)
        assert stats.error_of_mean == pytest.approx(
            eloc.mean(axis=0).std(ddof=0) / np.sqrt(n_chains)
        )
        # variance == average sample variance over chains
        assert stats.variance == pytest.approx(eloc.var(axis=0).mean())
        # R estimate
        B_over_n = stats.error_of_mean ** 2
        W = stats.variance
        assert stats.R == pytest.approx(
            np.sqrt((n_samples - 1.0) / n_samples + B_over_n / W), abs=1e-3
        )


def test_stats_mean_std():
    hi, ham, ma = _setup()

    for bs in (1, 2, 16, 32):
        _test_stats_mean_std(hi, ham, ma, bs)
