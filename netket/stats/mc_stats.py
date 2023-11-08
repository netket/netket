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
from typing import Union

import jax
import numpy as np
from jax import numpy as jnp

from netket.utils import config, mpi, struct
from netket.jax.sharding import extract_replicated

from . import mean as _mean
from . import var as _var
from . import total_size as _total_size
from ._autocorr import integrated_time


def _format_decimal(value, std, var):
    if math.isfinite(std) and std > 1e-7:
        decimals = max(int(np.ceil(-np.log10(std))), 0)
        return (
            "{0:.{1}f}".format(value, decimals + 1),
            "{0:.{1}f}".format(std, decimals + 1),
            "{0:.{1}f}".format(var, decimals + 1),
        )
    else:
        return (
            f"{value:.3e}",
            f"{std:.3e}",
            f"{var:.3e}",
        )


_NaN = float("NaN")


def _maybe_item(x):
    if hasattr(x, "shape") and x.shape == ():
        return x.item()
    else:
        return x


@struct.dataclass
class Stats:
    """A dict-compatible pytree containing the result of the statistics function."""

    mean: Union[float, complex] = _NaN
    """The mean value."""
    error_of_mean: float = _NaN
    """Estimate of the error of the mean."""
    variance: float = _NaN
    """Estimation of the variance of the data."""
    tau_corr: float = _NaN
    """Estimate of the autocorrelation time (in dimensionless units of number of steps).

    This value is estimated with a blocking algorithm by default, but the result is known
    to be unreliable. A more precise estimator based on the FFT transform can be used by
    setting the environment variable `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION=1`. This
    estimator is more computationally expensive, but overall the added cost should be
    negligible.
    """
    R_hat: float = _NaN
    """
    Estimator of the split-Rhat convergence estimator.

    The split-Rhat diagnostic is based on comparing intra-chain and inter-chain
    statistics of the sample and is thus only available for 2d-array inputs where
    the rows are independently sampled MCMC chains. In an ideal MCMC samples,
    R_hat should be 1.0. If it deviates from this value too much, this indicates
    MCMC convergence issues. Thresholds such as R_hat > 1.1 or even R_hat > 1.01 have
    been suggested in the literature for when to discard a sample. (See, e.g.,
    Gelman et al., `Bayesian Data Analysis <http://www.stat.columbia.edu/~gelman/book/>`_,
    or Vehtari et al., `arXiv:1903.08008 <https://arxiv.org/abs/1903.08008>`_.)
    """
    tau_corr_max: float = _NaN
    """
    Estimate of the maximum autocorrelation time among all Markov chains.

    This value is only computed if the environment variable
    `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION` is set.
    """

    __module__ = "netket.stats"

    def to_dict(self):
        jsd = {}
        jsd["Mean"] = _maybe_item(self.mean)
        jsd["Variance"] = _maybe_item(self.variance)
        jsd["Sigma"] = _maybe_item(self.error_of_mean)
        jsd["R_hat"] = _maybe_item(self.R_hat)
        jsd["TauCorr"] = _maybe_item(self.tau_corr)
        if config.netket_experimental_fft_autocorrelation:
            jsd["TauCorrMax"] = _maybe_item(self.tau_corr_max)
        return jsd

    def to_compound(self):
        return "Mean", self.to_dict()

    def __repr__(self):
        # extract adressable data from fully replicated arrays
        self = extract_replicated(self)
        mean, err, var = _format_decimal(self.mean, self.error_of_mean, self.variance)
        if not math.isnan(self.R_hat):
            ext = f", R̂={self.R_hat:.4f}"
        else:
            ext = ""
        if config.netket_experimental_fft_autocorrelation:
            if not (math.isnan(self.tau_corr) and math.isnan(self.tau_corr_max)):
                ext += f", τ={self.tau_corr:.1f}<{self.tau_corr_max:.1f}"
        return f"{mean} ± {err} [σ²={var}{ext}]"

    # Alias accessors
    def __getattr__(self, name):
        if name in ("mean", "Mean"):
            return self.mean
        elif name in ("variance", "Variance"):
            return self.variance
        elif name in ("error_of_mean", "Sigma"):
            return self.error_of_mean
        elif name in ("R_hat", "R"):
            return self.R_hat
        elif name in ("tau_corr", "TauCorr"):
            return self.tau_corr
        elif name in ("tau_corr_max", "TauCorrMax"):
            return self.tau_corr_max
        else:
            raise AttributeError(f"'Stats' object object has no attribute '{name}'")

    def real(self):
        return self.replace(mean=np.real(self.mean))

    def imag(self):
        return self.replace(mean=np.imag(self.mean))


def _get_blocks(data, block_size):
    chain_length = data.shape[1]

    n_blocks = int(np.floor(chain_length / float(block_size)))

    return data[:, 0 : n_blocks * block_size].reshape((-1, block_size)).mean(axis=1)


def _block_variance(data, l):
    blocks = _get_blocks(data, l)
    ts = _total_size(blocks)
    if ts > 0:
        return _var(blocks), ts
    else:
        return jnp.nan, 0


def _batch_variance(data):
    b_means = data.mean(axis=1)
    ts = _total_size(b_means)
    return _var(b_means), ts


def _split_R_hat(data, W):
    N = data.shape[-1]
    if not config.netket_use_plain_rhat:
        # compute split-chain batch variance
        local_batch_size = data.shape[0]
        if N % 2 == 0:
            # split each chain in the middle,
            # like [[1 2 3 4]] -> [[1 2][3 4]]
            batch_var, _ = _batch_variance(data.reshape(2 * local_batch_size, N // 2))
        else:
            # drop the last sample of each chain for an even split,
            # like [[1 2 3 4 5]] -> [[1 2][3 4]]
            batch_var, _ = _batch_variance(
                data[:, :-1].reshape(2 * local_batch_size, N // 2)
            )

    # V_loc = _np.var(data, axis=-1, ddof=0)
    # W_loc = _np.mean(V_loc)
    # W = _mean(W_loc)
    # # This approximation seems to hold well enough for larger n_samples
    return jnp.sqrt((N - 1) / N + batch_var / W)


BLOCK_SIZE = 32


def statistics(data):
    r"""
    Returns statistics of a given array (or matrix, see below) containing a stream of data.
    This is particularly useful to analyze Markov Chain data, but it can be used
    also for other type of time series.
    Assumes same shape on all MPI processes.

    Args:
        data (vector or matrix): The input data. It can be real or complex valued.
            * if a vector, it is assumed that this is a time series of data (not necessarily independent);
            * if a matrix, it is assumed that that rows :code:`data[i]` contain independent time series.

    Returns:
       Stats:
        A dictionary-compatible class containing the
        average (:code:`.mean`, :code:`["Mean"]`),
        variance (:code:`.variance`, :code:`["Variance"]`),
        the Monte Carlo standard error of the mean (:code:`error_of_mean`, :code:`["Sigma"]`),
        an estimate of the autocorrelation time (:code:`tau_corr`, :code:`["TauCorr"]`), and the
        Gelman-Rubin split-Rhat diagnostic (:code:`.R_hat`, :code:`["R_hat"]`).

        If the flag `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION` is set, the autocorrelation is computed
        exactly using a FFT transform, and an extra field `tau_corr_max` is inserted in the
        statistics object

        These properties can be accessed both the attribute and the dictionary-style syntax
        (both indicated above).

        The split-Rhat diagnostic is based on comparing intra-chain and inter-chain
        statistics of the sample and is thus only available for 2d-array inputs where
        the rows are independently sampled MCMC chains. In an ideal MCMC samples,
        R_hat should be 1.0. If it deviates from this value too much, this indicates
        MCMC convergence issues. Thresholds such as R_hat > 1.1 or even R_hat > 1.01 have
        been suggested in the literature for when to discard a sample. (See, e.g.,
        Gelman et al., `Bayesian Data Analysis <http://www.stat.columbia.edu/~gelman/book/>`_,
        or Vehtari et al., `arXiv:1903.08008 <https://arxiv.org/abs/1903.08008>`_.)
    """
    if config.netket_experimental_fft_autocorrelation:
        return _statistics(data)
    else:
        from .mc_stats_old import statistics as statistics_blocks

        return statistics_blocks(data)


@jax.jit
def _statistics(data):
    data = jnp.atleast_1d(data)
    if data.ndim == 1:
        data = data.reshape((1, -1))

    if data.ndim > 2:
        raise NotImplementedError("Statistics are implemented only for ndim<=2")

    mean = _mean(data)
    variance = _var(data)

    taus = jax.vmap(integrated_time)(data)
    tau_avg, _ = mpi.mpi_mean_jax(jnp.mean(taus))
    tau_max, _ = mpi.mpi_max_jax(jnp.max(taus))

    batch_var, n_batches = _batch_variance(data)
    if n_batches > 1:
        error_of_mean = jnp.sqrt(batch_var / n_batches)
        R_hat = _split_R_hat(data, variance)
    else:
        l_block = max(1, data.shape[1] // BLOCK_SIZE)
        block_var, n_blocks = _block_variance(data, l_block)
        error_of_mean = jnp.sqrt(block_var / n_blocks)
        R_hat = jnp.nan

    res = Stats(mean, error_of_mean, variance, tau_avg, R_hat, tau_max)

    return res
