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

from .. import common

pytestmark = common.skipif_mpi


WEIGHT_SEED = 3


@partial(jax.jit, static_argnums=0)
@partial(jax.vmap, in_axes=(None, None, 0, 0, 0), out_axes=(0))
def local_value_kernel(logpsi, pars, σ, σp, mel):
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_values(logpsi, variables, Ô, σ):
    σp, mels = Ô.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))
    loc_vals = local_value_kernel(
        logpsi, variables, σ.reshape((-1, σ.shape[-1])), σp, mels
    )
    return loc_vals.reshape(σ.shape[:-1])


def _setup():
    g = nk.graph.Hypercube(3, 2)
    hi = nk.hilbert.Spin(0.5, N=g.n_nodes)

    ham = nk.operator.Heisenberg(hi, graph=g)

    ma = nk.models.RBM(alpha=2, dtype=np.complex64)

    return hi, ham, ma


def _test_stats_mean_std(hi, ham, ma, n_chains):
    w = ma.init(jax.random.PRNGKey(WEIGHT_SEED * n_chains), jnp.zeros((1, hi.size)))

    sampler = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)

    n_samples = 16000
    num_samples_per_chain = n_samples // n_chains

    # Discard a few samples
    _, state = sampler.sample(ma, w, chain_length=1000)

    samples, state = sampler.sample(
        ma, w, chain_length=num_samples_per_chain, state=state
    )
    assert samples.shape == (num_samples_per_chain, n_chains, hi.size)

    # eloc = np.empty((num_samples_per_chain, n_chains), dtype=np.complex128)
    eloc = local_values(ma.apply, w, ham, samples)
    assert eloc.shape == (num_samples_per_chain, n_chains)

    stats = statistics(eloc.T)

    assert stats.mean == pytest.approx(np.mean(eloc))
    if n_chains > 1:
        # variance == average sample variance over chains
        assert stats.variance == pytest.approx(np.var(eloc))


@common.skipif_mpi
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
        return -(x**2.0) / 2.0

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


@common.skipif_mpi
def test_R_hat():
    # detect disagreeing chains
    x = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
        ]
    )
    assert statistics(x).R_hat > 1.01

    # detect non-stationary chains
    x = np.array(
        [
            [1.0, 1.5, 2.0],
            [2.0, 1.5, 1.0],
        ]
    )
    assert statistics(x).R_hat > 1.01

    # detect "stuck" chains
    x = np.array(
        [
            np.random.normal(size=1000),
            np.random.normal(size=1000),
        ]
    )
    # not stuck -> good R_hat:
    assert statistics(x).R_hat <= 1.01
    # stuck -> bad  R_hat:
    x[1, 100:] = 1.0
    assert statistics(x).R_hat > 1.01
