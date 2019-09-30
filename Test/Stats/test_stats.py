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


def _test_stats_mean_std(hi, ham, ma, batch_size):
    sampler = nk.sampler.MetropolisLocal(ma, batch_size=batch_size)

    n_samples = 16000
    num_samples_per_chain = n_samples // batch_size

    samples, log_values = nk.sampler.compute_samples(
        sampler, n_samples=n_samples, n_discard=6400
    )
    assert samples.shape == (num_samples_per_chain, batch_size, hi.size)

    eloc = local_values(ham, ma, samples, log_values)
    assert eloc.shape == (num_samples_per_chain, batch_size)

    stats = statistics(eloc)

    # These tests only work for one MPI process
    assert nk.MPI.size() == 1

    assert stats.mean == pytest.approx(np.mean(eloc))
    if batch_size > 1:
        # error of mean == stdev of sample mean between chains / sqrt(#chains)
        assert stats.error_of_mean == pytest.approx(
            eloc.mean(axis=0).std(ddof=0) / np.sqrt(batch_size)
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
