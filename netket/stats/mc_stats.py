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

from functools import partial

import jax
from jax import numpy as jnp

import numpy as np

from netket.utils import mpi
from netket.utils import struct

from .mpi_stats import var as _var, sum as _sum
from .stats_base import StatsBase


@struct.dataclass
class ChainStats(StatsBase):
    """A Statistics class that uses the variance of the chains to compute
    correlation time and Rhat.
    """

    @struct.property_cached(pytree_node=True)
    def error(self) -> float:
        return jnp.sqrt(self.batch_variance / self.n_chains)

    @struct.property_cached(pytree_node=True)
    def r_hat(self) -> float:
        if self.n_chains > 1:
            return jnp.sqrt(
                (self._n - 1) / self._n + self.batch_variance / self.variance
            )
        else:
            return jnp.nan

    @struct.property_cached(pytree_node=True)
    def tau_corr(self) -> float:
        return jnp.clip(0.5 * (self._n * self.batch_variance / self.variance - 1), 0)

    @struct.property_cached(pytree_node=True)
    def batch_variance(self) -> float:
        return _var(self._μ)

    def merge(self, other):
        """
        Merge the two statistics.
        """
        return super().merge(other)

    def __repr__(self):
        return super().__repr__()


# TODO: Find a better algorithm.
# This blocked versions does not perform very well and has a high error
@struct.dataclass
class BlockStats(StatsBase):
    """A Statistics class that uses re-blocking to compute correlation time.
    Rhat is 0.
    """

    _M2_block: float = 0.0
    """Sum of the squared residuals of the blocks."""
    _n_blocks: int = 0
    """The number of blocks in the reduced data."""
    _block_size: int = struct.field(pytree_node=False, default=0)
    """The size of different blocks."""

    @property
    def n_blocks(self):
        return self._n_blocks * mpi.n_nodes

    @struct.property_cached(pytree_node=True)
    def error(self) -> float:
        return jnp.sqrt(self.block_variance / self.n_blocks)

    @struct.property_cached(pytree_node=True)
    def r_hat(self) -> float:
        return jnp.nan

    @struct.property_cached(pytree_node=True)
    def tau_corr(self) -> float:
        return jnp.clip(
            0.5 * (self._block_size * self.block_variance / self.variance - 1), 0
        )

    @struct.property_cached(pytree_node=True)
    def block_variance(self) -> float:
        n = self._n_blocks
        B = mpi.n_nodes

        if B > 1:
            μ_block = self._μ.mean()
            # This is derived from the Weldford's formula for merging two M2, assuming
            # they all have the same number of observations n
            M2 = _sum(
                self._M2_block + n * (μ_block * (μ_block - self.mean).conj()).real
            )
        else:
            M2 = self._M2_block
        return M2 / (B * n)

    def merge(self, other):
        """
        Merge the two statistics.
        """
        if self._block_size != other._block_size:
            raise ValueError("Block sizes must match.")

        self = super().merge(other)

        n_blocks = self._n_blocks + other._n_blocks

        δ = other._μ.mean() - self._μ.mean()
        M2_block = (
            self._M2_block
            + other._M2_block
            + (jnp.abs(δ) ** 2) * (self._n_blocks * other._n_blocks) / n_blocks
        )

        return self.replace(_M2_block=M2_block, _n_blocks=n_blocks)

    def __repr__(self):
        return super().__repr__()


def _get_blocks(data, block_size):
    chain_length = data.shape[1]

    n_blocks = int(np.floor(chain_length / float(block_size)))

    return (
        data[:, 0 : n_blocks * block_size]
        .reshape((-1, n_blocks, block_size))
        .mean(axis=-1)
    )


# this is not batch_size maybe?
def statistics(data, block_size=32, precompute=True) -> StatsBase:
    r"""
    Returns statistics of a given array (or matrix, see below) containing a stream of data.
    This is particularly useful to analyze Markov Chain data, but it can be used
    also for other type of time series.
    Assumes same shape on all MPI processes.

    Args:
        data (vector or matrix): The input data. It can be real or complex valued.
                                * if a vector, it is assumed that this is a time
                                  series of data (not necessarily independent).
                                * if a matrix, it is assumed that that rows data[i]
                                  contain independent time series.

    Returns:
       Stats: A dictionary-compatible class containing the average (mean),
             the variance (variance),
             the error of the mean (error_of_mean), and an estimate of the
             autocorrelation time (tau_corr). In addition to accessing the elements with the standard
             dict sintax (e.g. res['mean']), one can also access them directly with the dot operator
             (e.g. res.mean).
    """
    return _statistics(data, block_size, precompute)


@partial(jax.jit, static_argnums=(1, 2))
def _statistics(data, block_size, precompute):
    data = jnp.atleast_1d(data)
    if data.ndim == 1:
        data = data.reshape((1, -1))

    if data.ndim > 2:
        raise NotImplementedError("Statistics are implemented only for ndim<=2")

    μ = data.mean(axis=-1)
    n = data.shape[-1]
    M2 = data.var(axis=-1) * n

    # use blocking algorithm instead of batches if less than 8 chains
    use_blocks = mpi.n_nodes * μ.shape[0] < 8
    if not use_blocks:
        res = ChainStats(
            _μ=μ,
            _M2=M2,
            _n=n,
            __precompute_cached_properties=precompute,
        )

    else:
        chain_length = data.shape[-1]

        # compute the total number of blocks, making sure that there are at least 32 blocks
        # and special case the condition where the chain is shorter than 32 elements
        n_blocks_min = min(32, chain_length)
        n_blocks = max(n_blocks_min, int(np.floor(chain_length / float(block_size))))

        # Compute the actual block size starting from the total number of blocks
        block_size = max(1, chain_length // n_blocks)
        data = _get_blocks(data, block_size)

        n_blocks = data.size
        M2_block = data.var() * n_blocks

        res = BlockStats(
            _μ=μ,
            _M2=M2,
            _n=n,
            _n_blocks=n_blocks,
            _M2_block=M2_block,
            _block_size=block_size,
            __precompute_cached_properties=precompute,
        )

    return res
    ##
