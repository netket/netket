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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from functools import partial

import netket as nk
from netket.stats import statistics
from scipy.optimize import curve_fit
from numba import jit

from netket.stats import mean as mpi_mean, var as mpi_var


@pytest.mark.parametrize("n_chains", [1, 2, 3, 8, 9, 16])
@pytest.mark.parametrize("dtype", [float, complex])
def test_stats_mean_std(n_chains, _mpi_size, dtype):
    n_samples = 10000
    num_samples_per_chain = n_samples // (n_chains * _mpi_size)

    seed = nk.jax.mpi_split(nk.jax.PRNGKey(0))
    # pick a random distribution ...
    data = jax.random.normal(seed, (num_samples_per_chain, n_chains), dtype=dtype)

    block_size = 32
    stats = nk.stats.statistics(data.T, block_size)

    assert stats.mean == pytest.approx(mpi_mean(data))
    if n_chains >= 8:
        assert isinstance(stats, nk.stats.ChainStats)
        # variance == average sample variance over chains
        assert stats.variance == pytest.approx(mpi_var(data))
        # R estimate
        B_over_n = stats.error_of_mean ** 2
        W = stats.variance
        assert stats.R_hat == pytest.approx(np.sqrt(1.0 + B_over_n / W), abs=1e-3)
    else:
        assert isinstance(stats, nk.stats.BlockStats)
        # it uses blocks
        n_blocks = (num_samples_per_chain * n_chains) // block_size
        assert n_blocks == stats.n_blocks

        data_blocks = data.reshape(-1)[: n_blocks * block_size]


@pytest.mark.parametrize("n_chains", [1, 2, 8, 16])
@pytest.mark.parametrize("dtype", [float, complex])
@pytest.mark.parametrize("n_splits", [2, 3, 6])
def test_stats_merge(n_chains, n_splits, dtype, _mpi_size):
    n_samples_tot = 1000000
    num_samples_per_chain = n_samples_tot // (n_chains * _mpi_size * n_splits)

    seed = nk.jax.mpi_split(nk.jax.PRNGKey(0))
    # pick a random distribution ...
    data = jax.random.normal(
        seed, (n_splits, num_samples_per_chain, n_chains), dtype=dtype
    )

    data_1 = data.reshape(-1, n_chains)
    stats = nk.stats.statistics(data_1.T)

    stats_acc = nk.stats.statistics(data[0].T)
    for i in range(1, n_splits):
        stats_acc = stats_acc.merge(nk.stats.statistics(data[i].T))

    if isinstance(stats_acc, nk.stats.BlockStats):
        err_rtol = 6e-2
        tau_rtol = 5e-2
        tau_atol = 2e-2
    else:
        err_rtol = 1e-5
        tau_rtol = 1e-5
        tau_atol = 0

    np.testing.assert_allclose(stats_acc.mean, stats.mean)
    np.testing.assert_allclose(stats_acc.variance, stats.variance)
    np.testing.assert_allclose(stats_acc.error, stats.error, rtol=err_rtol)
    np.testing.assert_allclose(stats_acc.r_hat, stats.r_hat)
    np.testing.assert_allclose(
        stats_acc.tau_corr, stats.tau_corr, rtol=tau_rtol, atol=tau_atol
    )


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16, 32, 64])
def test_tau_corr(batch_size):
    sig_corr = 0.5

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

    assert mpi_mean(data) == pytest.approx(stats.mean)
    assert mpi_var(data) == pytest.approx(stats.variance)

    assert tau_fit_m == pytest.approx(stats.tau_corr, rel=1, abs=3)

    eom_fit = np.sqrt(mpi_var(data) * tau_fit_m / float(n_samples * batch_size))

    print(stats.error_of_mean, eom_fit)
    assert eom_fit == pytest.approx(stats.error_of_mean, rel=0.6)


def test_decimal_format():
    from netket.stats.stats_base import _format_decimal

    _format_decimal(1.0, 1e-3, np.nan) == ("1.0000", "0.0010", "nan")
    _format_decimal(1.0, 1e-6, np.nan) == ("1.0000000", "0.0000010", "nan")
    _format_decimal(1.0, 1e-7, np.nan) == ("1.000e+00", "1.000e-07", "nan")

    _format_decimal(float("nan"), float("inf"), np.nan) == ("nan", "inf", "nan")
    _format_decimal(1.0, float("nan"), np.nan) == ("1.000e+00", "nan", "nan")
    _format_decimal(1.0, float("inf"), np.nan) == ("1.000e+00", "inf", "nan")
    _format_decimal(float("inf"), 0.0, np.nan) == ("inf", "0.000e+00", "nan")
    _format_decimal(1.0, 0.0, np.nan) == ("1.000e+00", "0.000e+00", "nan")

    _format_decimal(1.0, 0.12, 0.5) == ("1.00", "0.12", "0.50")
