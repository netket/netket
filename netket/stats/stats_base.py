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

from typing import Union
import math

import numpy as np

import jax.numpy as jnp

from netket.utils import mpi
from netket.utils import struct

from .mpi_stats import mean as _mean, sum as _sum


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


@struct.dataclass
class StatsBase:
    """A dict-compatible class containing the result of the statistics function.

    This class should be derived to implement rhat and tau_corr."""

    _μ: Union[float, complex] = 0
    """The mean value per-chain."""
    _M2: float = 0
    """The sum of the residuals squared, per chain."""
    _n: int = 0
    """The number of datapoints per chain."""

    @property
    def n_chains(self):
        """The total number of independent streams in the data."""
        return self._μ.size * mpi.n_nodes

    @struct.property_cached(pytree_node=True)
    def mean(self) -> float:
        """The sample mean of the reduced data."""
        return _mean(self._μ)

    @struct.property_cached(pytree_node=True)
    def variance(self) -> float:
        """The sample variance of the reduced data."""
        n = self._n
        B = self.n_chains

        # This is derived from the Weldford's formula for merging two M2, assuming
        # they all have the same number of observations n
        M2 = _sum(self._M2 + n * (self._μ * (self._μ - self.mean).conj()).real)
        return M2 / (B * n)

    def merge(self, other):
        """
        Merge two statistics objects.
        """
        if self.n_chains != other.n_chains:
            raise ValueError(
                "Cannot sum two Stats objects for a different number of chains."
            )

        n = self._n + other._n

        a1 = self._n / n
        a2 = other._n / n

        μ = a1 * self._μ + a2 * other._μ

        δ = other._μ - self._μ
        M2 = self._M2 + other._M2 + (jnp.abs(δ) ** 2) * (self._n * other._n) / n

        return self.replace(_μ=μ, _M2=M2, _n=n)

    def __repr__(self):
        mean, err, var = _format_decimal(self.mean, self.error, self.variance)
        if not math.isnan(self.r_hat):
            ext = ", R̂={:.4f}".format(self.r_hat)
        else:
            ext = ""
        return "{} ± {} [σ²={}{}]".format(mean, err, var, ext)

    def to_dict(self):
        jsd = {}
        jsd["Mean"] = self.mean.item()
        jsd["Variance"] = self.variance.item()
        jsd["Sigma"] = self.error.item()
        jsd["Rhat"] = self.r_hat.item()
        jsd["TauCorr"] = self.tau_corr.item()
        return jsd

    def to_compound(self):
        return "Mean", self.to_dict()

    # Alias accessors
    def __getattr__(self, name):
        if name in ("mean", "Mean"):
            return self.mean
        elif name in ("variance", "Variance"):
            return self.variance
        elif name in ("error_of_mean", "Sigma"):
            return self.error
        elif name in ("R_hat", "R"):
            return self.r_hat
        elif name in ("tau_corr", "TauCorr"):
            return self.tau_corr
        else:
            raise AttributeError(
                "'Stats' object object has no attribute '{}'".format(name)
            )
