import numpy as np
import pytest

import netket.legacy as nk
import netket.variational as vmc
from netket.operator import local_values
from netket.stats import statistics
from scipy.optimize import curve_fit
from numba import jit


def _setup():
    g = nk.graph.Hypercube(3, 2)
    hi = nk.hilbert.Spin(0.5, N=g.n_nodes)

    ham = nk.operator.Heisenberg(hi, graph=g)

    ma = nk.machine.RbmSpin(hi, alpha=2)
    ma.init_random_parameters()

    return hi, ham, ma


def _test_stats_mean_std(hi, ham, ma, n_chains):
    sampler = nk.sampler.MetropolisLocal(ma, n_chains=n_chains)

    n_samples = 16000
    num_samples_per_chain = n_samples // n_chains

    # Discard a few samples
    sampler.generate_samples(1000)

    samples = sampler.generate_samples(num_samples_per_chain)
    assert samples.shape == (num_samples_per_chain, n_chains, hi.size)

    eloc = np.empty((num_samples_per_chain, n_chains), dtype=np.complex128)
    for i in range(num_samples_per_chain):
        eloc[i] = local_values(ham, ma, samples[i])

    stats = statistics(eloc.T)

    # These tests only work for one MPI process
    assert nk.stats.MPI.COMM_WORLD.size == 1

    assert stats.mean == pytest.approx(np.mean(eloc))
    if n_chains > 1:

        # variance == average sample variance over chains
        assert stats.variance == pytest.approx(np.var(eloc))
        # R estimate
        B_over_n = stats.error_of_mean ** 2
        W = stats.variance
        assert stats.R_hat == pytest.approx(np.sqrt(1.0 + B_over_n / W), abs=1e-3)


def test_stats_mean_std():
    hi, ham, ma = _setup()
    for bs in (1, 2, 16, 32):
        _test_stats_mean_std(hi, ham, ma, bs)


def _test_tau_corr(batch_size, sig_corr):
    def next_pow_two(n):
        i = 1
        while i < n:
            i = i << 1
        return i

    def autocorr_func_1d(x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n

        # Optionally normalize
        if norm:
            acf /= acf[0]

        return acf

    @jit
    def gen_data(n_samples, log_f, dx, seed=1234):
        np.random.seed(seed)
        # Generates data with a simple markov chain
        x = np.empty(n_samples)
        x_old = np.random.normal()
        for i in range(n_samples):
            x_new = x_old + np.random.normal(scale=dx, loc=0.0)
            if np.exp(log_f(x_new) - log_f(x_old)) > np.random.uniform(0, 1):
                x[i] = x_new
            else:
                x[i] = x_old

            x_old = x[i]
        return x

    @jit
    def log_f(x):
        return -(x ** 2.0) / 2.0

    def func_corr(x, tau):
        return np.exp(-x / (tau))

    n_samples = 8000000 // batch_size

    data = np.empty((batch_size, n_samples))
    tau_fit = np.empty((batch_size))

    for i in range(batch_size):
        data[i] = gen_data(n_samples, log_f, sig_corr, seed=i + batch_size)
        autoc = autocorr_func_1d(data[i])
        popt, pcov = curve_fit(func_corr, np.arange(40), autoc[0:40])
        tau_fit[i] = popt[0]

    tau_fit_m = tau_fit.mean()

    stats = statistics(data)

    assert np.mean(data) == pytest.approx(stats.mean)
    assert np.var(data) == pytest.approx(stats.variance)

    assert tau_fit_m == pytest.approx(stats.tau_corr, rel=1, abs=3)

    eom_fit = np.sqrt(np.var(data) * tau_fit_m / float(n_samples * batch_size))

    print(stats.error_of_mean, eom_fit)
    assert eom_fit == pytest.approx(stats.error_of_mean, rel=0.6)


def test_tau_corr():
    sig_corr = 0.5
    for bs in (1, 2, 32, 64):
        _test_tau_corr(bs, sig_corr)


def test_decimal_format():
    from netket.stats import Stats

    assert str(Stats(1.0, 1e-3)) == "1.0000 ± 0.0010 [σ²=nan]"
    assert str(Stats(1.0, 1e-6)) == "1.0000000 ± 0.0000010 [σ²=nan]"
    assert str(Stats(1.0, 1e-7)) == "1.000e+00 ± 1.000e-07 [σ²=nan]"

    assert str(Stats(float("nan"), float("inf"))) == "nan ± inf [σ²=nan]"
    assert str(Stats(1.0, float("nan"))) == "1.000e+00 ± nan [σ²=nan]"
    assert str(Stats(1.0, float("inf"))) == "1.000e+00 ± inf [σ²=nan]"
    assert str(Stats(float("inf"), 0.0)) == "inf ± 0.000e+00 [σ²=nan]"
    assert str(Stats(1.0, 0.0)) == "1.000e+00 ± 0.000e+00 [σ²=nan]"

    assert str(Stats(1.0, 0.12, 0.5)) == "1.00 ± 0.12 [σ²=0.50]"
    assert str(Stats(1.0, 0.12, 0.5, R_hat=1.01)) == "1.00 ± 0.12 [σ²=0.50, R̂=1.0100]"
