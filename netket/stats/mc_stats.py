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

from functools import partial

from flax import struct
import jax
from jax import numpy as jnp

import numpy as np

from netket import jax as nkjax
from netket.utils import config

from . import mean as _mean
from . import var as _var
from . import total_size as _total_size


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
            "{0:.3e}".format(value),
            "{0:.3e}".format(std),
            "{0:.3e}".format(var),
        )


_NaN = float("NaN")


@struct.dataclass
class Stats:
    """A dict-compatible class containing the result of the statistics function."""

    mean: Union[float, complex] = _NaN
    """The mean value"""
    error_of_mean: float = _NaN
    variance: float = _NaN
    tau_corr: float = _NaN
    R_hat: float = _NaN

    def to_dict(self):
        jsd = {}
        jsd["Mean"] = self.mean.item()
        jsd["Variance"] = self.variance.item()
        jsd["Sigma"] = self.error_of_mean.item()
        jsd["R_hat"] = self.R_hat.item()
        jsd["TauCorr"] = self.tau_corr.item()
        return jsd

    def to_compound(self):
        return "Mean", self.to_dict()

    def __repr__(self):
        mean, err, var = _format_decimal(self.mean, self.error_of_mean, self.variance)
        if not math.isnan(self.R_hat):
            ext = ", R̂={:.4f}".format(self.R_hat)
        else:
            ext = ""
        return "{} ± {} [σ²={}{}]".format(mean, err, var, ext)

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
        else:
            raise AttributeError(
                "'Stats' object object has no attribute '{}'".format(name)
            )


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


# this is not batch_size maybe?
def statistics(data, batch_size=32):
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
       Stats: A dictionary-compatible class containing the
             average (:code:`.mean`, :code:`["Mean"]`),
             variance (:code:`.variance`, :code:`["Variance"]`),
             the Monte Carlo standard error of the mean (:code:`error_of_mean`, :code:`["Sigma"]`),
             an estimate of the autocorrelation time (:code:`tau_corr`, :code:`["TauCorr"]`), and the
             Gelman-Rubin split-Rhat diagnostic (:code:`.R_hat`, :code:`["R_hat"]`).

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
    return _statistics(data, batch_size)


@partial(jax.jit, static_argnums=1)
def _statistics(data, batch_size):
    data = jnp.atleast_1d(data)
    if data.ndim == 1:
        data = data.reshape((1, -1))

    if data.ndim > 2:
        raise NotImplementedError("Statistics are implemented only for ndim<=2")

    mean = _mean(data)
    variance = _var(data)

    ts = _total_size(data)

    bare_var = variance

    batch_var, n_batches = _batch_variance(data)

    l_block = max(1, data.shape[1] // batch_size)

    block_var, n_blocks = _block_variance(data, l_block)

    tau_batch = ((ts / n_batches) * batch_var / bare_var - 1) * 0.5
    tau_block = ((ts / n_blocks) * block_var / bare_var - 1) * 0.5

    batch_good = (tau_batch < 6 * data.shape[1]) * (n_batches >= batch_size)
    block_good = (tau_block < 6 * l_block) * (n_blocks >= batch_size)

    stat_dtype = nkjax.dtype_real(data.dtype)

    # if batch_good:
    #    error_of_mean = jnp.sqrt(batch_var / n_batches)
    #    tau_corr = jnp.max(0, tau_batch)
    # elif block_good:
    #    error_of_mean = jnp.sqrt(block_var / n_blocks)
    #    tau_corr = jnp.max(0, tau_block)
    # else:
    #    error_of_mean = jnp.nan
    #    tau_corr = jnp.nan
    # jax style

    def batch_good_err(args):
        batch_var, tau_batch, *_ = args
        error_of_mean = jnp.sqrt(batch_var / n_batches)
        tau_corr = jnp.clip(tau_batch, 0)
        return jnp.asarray(error_of_mean, dtype=stat_dtype), jnp.asarray(
            tau_corr, dtype=stat_dtype
        )

    def block_good_err(args):
        _, _, block_var, tau_block = args
        error_of_mean = jnp.sqrt(block_var / n_blocks)
        tau_corr = jnp.clip(tau_block, 0)
        return jnp.asarray(error_of_mean, dtype=stat_dtype), jnp.asarray(
            tau_corr, dtype=stat_dtype
        )

    def nan_err(args):
        return jnp.asarray(jnp.nan, dtype=stat_dtype), jnp.asarray(
            jnp.nan, dtype=stat_dtype
        )

    def batch_not_good(args):
        batch_var, tau_batch, block_var, tau_block, block_good = args
        return jax.lax.cond(
            block_good,
            block_good_err,
            nan_err,
            (batch_var, tau_batch, block_var, tau_block),
        )

    error_of_mean, tau_corr = jax.lax.cond(
        batch_good,
        batch_good_err,
        batch_not_good,
        (batch_var, tau_batch, block_var, tau_block, block_good),
    )

    if n_batches > 1:
        N = data.shape[-1]

        if not config.FLAGS["NETKET_USE_PLAIN_RHAT"]:
            # compute split-chain batch variance
            local_batch_size = data.shape[0]
            if N % 2 == 0:
                # split each chain in the middle,
                # like [[1 2 3 4]] -> [[1 2][3 4]]
                batch_var, _ = _batch_variance(
                    data.reshape(2 * local_batch_size, N // 2)
                )
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
        W = variance

        R_hat = jnp.sqrt((N - 1) / N + batch_var / W)
    else:
        R_hat = jnp.nan

    res = Stats(mean, error_of_mean, variance, tau_corr, R_hat)

    return res
    ##
