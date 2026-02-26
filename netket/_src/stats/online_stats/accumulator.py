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

"""OnlineStats: streaming MCMC accumulator (Welford + online ACF)."""

from math import isnan, sqrt

import numpy as np
import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.stats.mc_stats import Stats

from netket._src.stats.online_stats.kernels import _update_arrays

_NaN = jnp.nan


class OnlineStats(struct.Pytree):
    """Streaming accumulator for MCMC statistics across multiple batches.

    Accumulates mean, variance, tau_corr, R_hat, and error_of_mean
    incrementally using the parallel Welford algorithm. Supports optional
    exponential decay (EMA) to down-weight old data.

    When ``max_lag > 0`` (default 64), the autocovariance function at lags
    ``0..max_lag`` is also tracked online via a per-chain sample buffer.
    This enables a Geyer IPS+IMS estimate of the integrated autocorrelation
    time (``tau_corr_acf``) that works even with a single chain.

    All per-chain arrays are JAX arrays (``pytree_node=True``) so that the
    object is a valid JAX pytree.  The scalar configuration fields (``decay``,
    ``max_lag``) and the Python-int counters (``_n_samples_total``,
    ``_buf_len``) are static (``pytree_node=False``).

    Use :func:`online_statistics` as the functional API, or
    :meth:`from_data` to construct directly from a first batch.

    Example::

        estimator = OnlineStats.from_data(first_batch)
        for batch in remaining_batches:
            estimator = estimator.update(batch)
        stats = estimator.get_stats()
    """

    # ---- Configuration ----
    max_lag: int = struct.field(pytree_node=False)
    _decay: float | None = struct.field(pytree_node=True)
    # we store _decay as a private field because we have a decay property which is
    # statically 1 if decay is None.

    # ---- Per-chain Welford state (JAX arrays, pytree leaves) ----
    # _chain_count : (n_chains,) float64  — effective sample count per chain
    # _chain_mean  : (n_chains,) dtype    — running mean, preserves input dtype
    # _chain_M2    : (n_chains,) float64  — running sum of squared deviations
    _chain_count: jax.Array = struct.field(pytree_node=True)
    _chain_mean: jax.Array = struct.field(pytree_node=True)
    _chain_M2: jax.Array = struct.field(pytree_node=True)

    # ---- Online ACF state (JAX arrays, pytree leaves) ----
    # Shapes: (n_chains, max_lag+1) for cross/m1/m2/pair;
    #         (n_chains, max_lag)   for chain_buf.
    # When max_lag==0 all shapes have 0 columns: (n_chains, 0).
    _cross_sum: jax.Array = struct.field(pytree_node=True)
    _m1_sum: jax.Array = struct.field(pytree_node=True)
    _m2_sum: jax.Array = struct.field(pytree_node=True)
    _pair_count: jax.Array = struct.field(pytree_node=True)
    _chain_buf: jax.Array = struct.field(pytree_node=True)

    # ---- Scalar counters ----
    _n_samples_total: int = struct.field(default=0)
    # _buf_len is a JAX integer (pytree leaf, not static) so that the gradual
    # fill-up of the rolling buffer does not trigger JIT recompilation.
    _buf_len: jax.Array = struct.field(pytree_node=True)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        n_chains: int,
        dtype,
        *,
        decay: float | None = None,
        max_lag: int = 64,
    ):
        """
        Initialize empty online statistics buffers.

        Args:
            n_chains: number of independent chains
            dtype: dtype of incoming samples (used for means)
            decay: EMA decay factor applied per update call
                (default None ≡ 1.0 → no decay)
            max_lag: maximum lag for online ACF estimator
                    (set 0 to disable)
        """
        max_lag = int(max_lag)
        acf_len = max_lag + 1 if max_lag > 0 else 0

        self.max_lag = max_lag
        self._decay = decay

        # per-chain running stats
        self._chain_count = jnp.zeros(n_chains, dtype=jnp.float64)
        self._chain_mean = jnp.zeros(n_chains, dtype=dtype)
        self._chain_M2 = jnp.zeros(n_chains, dtype=jnp.float64)

        # ACF accumulators
        self._cross_sum = jnp.zeros((n_chains, acf_len), dtype=jnp.float64)
        self._m1_sum = jnp.zeros((n_chains, acf_len), dtype=jnp.float64)
        self._m2_sum = jnp.zeros((n_chains, acf_len), dtype=jnp.float64)
        self._pair_count = jnp.zeros((n_chains, acf_len), dtype=jnp.float64)

        # lag buffer
        self._chain_buf = jnp.zeros((n_chains, max_lag), dtype=jnp.float64)
        self._buf_len = jnp.array(0, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_data(
        cls, data, *, decay: float | None = None, max_lag: int = 64
    ) -> "OnlineStats":
        """Construct an :class:`OnlineStats` from an initial batch of samples.

        Equivalent to calling :func:`online_statistics` with ``old_estimator=None``.

        Args:
            data: Array of shape ``(n_samples,)`` or ``(n_chains, n_samples_per_chain)``.
            decay: EMA decay factor (default None = no decay).
            max_lag: Maximum lag for the online ACF estimator (default 64).

        Returns:
            A new :class:`OnlineStats` initialized from *data*.
        """
        data = jnp.asarray(data)
        n_chains = 1 if data.ndim == 1 else data.shape[0]
        estimator = cls(n_chains, dtype=data.dtype, decay=decay, max_lag=max_lag)
        return estimator.update(data)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, data) -> "OnlineStats":
        """Incorporate a new batch of samples and return an updated :class:`OnlineStats`.

        Args:
            data: Array of shape ``(n_samples,)`` or
                ``(n_chains, n_samples_per_chain)``.  Must have the same
                number of chains as the existing accumulator.

        Returns:
            A new :class:`OnlineStats` with updated state.
        """
        data = jnp.asarray(data)
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2:
            raise ValueError(f"data must be 1D or 2D, got {data.ndim}D")

        n_chains_new, n_samples_per_chain = data.shape
        if n_chains_new != self.n_chains:
            raise ValueError(
                f"Number of chains changed: expected {self.n_chains}, got {n_chains_new}"
            )

        (
            chain_count,
            chain_mean,
            chain_M2,
            cross_sum,
            m1_sum,
            m2_sum,
            pair_count,
            chain_buf,
        ) = _update_arrays(
            self._chain_count,
            self._chain_mean,
            self._chain_M2,
            self._cross_sum,
            self._m1_sum,
            self._m2_sum,
            self._pair_count,
            self._chain_buf,
            self._decay,
            self.max_lag,
            self._buf_len,
            data,
        )

        new_buf_len = jnp.minimum(self._buf_len + n_samples_per_chain, self.max_lag)

        return self.replace(
            _chain_count=chain_count,
            _chain_mean=chain_mean,
            _chain_M2=chain_M2,
            _n_samples_total=self._n_samples_total + n_chains_new * n_samples_per_chain,
            _cross_sum=cross_sum,
            _m1_sum=m1_sum,
            _m2_sum=m2_sum,
            _pair_count=pair_count,
            _chain_buf=chain_buf,
            _buf_len=new_buf_len,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_chains(self) -> int:
        """Number of Markov chains."""
        return self._chain_mean.shape[0]

    @property
    def mean(self):
        """The current mean estimate (dtype matches the input data)."""
        total = jnp.sum(self._chain_count)
        if total == 0.0:
            return _NaN
        m = (jnp.sum(self._chain_count * self._chain_mean) / total).item()
        if isinstance(m, complex) and m.imag == 0.0:
            return m.real
        return m

    @property
    def variance(self):
        """The current variance estimate (float64)."""
        total = jnp.sum(self._chain_count)
        if total == 0.0:
            return _NaN
        global_mean = jnp.sum(self._chain_count * self._chain_mean) / total
        global_M2 = jnp.sum(self._chain_M2) + jnp.sum(
            self._chain_count * jnp.abs(self._chain_mean - global_mean) ** 2
        )
        return global_M2 / total

    @property
    def tau_corr(self):
        """Integrated autocorrelation time (ACF-based if available, else batch)."""
        tau = self.tau_corr_acf
        if isnan(tau):
            tau = self.tau_corr_batch
        return tau

    @property
    def tau_corr_batch(self):
        """Integrated autocorrelation time estimated from between-chain (batch) variance.

        For M independent Markov chains each of effective length n, the variance
        of the chain means B and the pooled within-chain variance W are related by:

        .. math::

            \\frac{B}{W} \\approx \\frac{1 + 2\\tau}{n}

        Rearranging gives the estimator:

        .. math::

            \\tau = \\frac{1}{2}\\left(\\frac{n \\, B}{W} - 1\\right)

        where:

        - :math:`n` = effective samples per chain (``n_samples_total / n_chains``)
        - :math:`B` = between-chain variance = ``Var(chain_means)``
        - :math:`W` = within-chain (pooled) variance = ``global_variance``

        Requires at least 2 chains and ``decay == 1.0``.  Returns ``NaN`` otherwise.
        """
        if self.n_chains < 2:
            return _NaN
        variance = self.variance
        if isnan(variance) or variance <= 0:
            return _NaN
        n_per_chain_eff = self._n_samples_total / self.n_chains
        B = jnp.var(jnp.real(self._chain_mean))
        return max((n_per_chain_eff * B / variance - 1) * 0.5, 0.0)

    @property
    def tau_corr_acf(self):
        """Integrated autocorrelation time from the online ACF via Geyer IPS+IMS.

        Uses Geyer's initial positive sequence (IPS): builds paired sums
        P[t] = rho[2t] + rho[2t+1] and discards all pairs from the first
        non-positive one onward.  Then enforces the initial monotone sequence
        (IMS) condition by taking a cumulative minimum.  Finally returns
        tau = 2 * sum(P) - 1.

        This is more robust than the Sokal window when max_lag is finite:
        the Sokal condition m >= c*tau requires m ~ 5*tau, which exceeds
        max_lag for any tau > max_lag/5, leading to inflated estimates.
        Geyer IPS truncates at the natural decay of the ACF regardless of tau.

        Returns ``NaN`` when ``max_lag == 0``, when ``decay != 1.0``, or
        before enough data has accumulated.
        """
        rho = self.acf
        if rho is None:
            return _NaN
        # Convert to NumPy for data-dependent indexing (outside JIT, so fine).
        rho_np = np.asarray(rho)
        m = len(rho_np) // 2
        if m == 0:
            return _NaN
        idx = 2 * np.arange(m)
        P = rho_np[idx] + rho_np[idx + 1]
        # IPS: truncate at first non-positive pair.
        nonpos = np.where(P <= 0)[0]
        if len(nonpos):
            P = P[: nonpos[0]]
        if len(P) == 0:
            return 1.0
        # IMS: enforce non-increasing (cumulative minimum).
        P = np.minimum.accumulate(P)
        return max(2.0 * np.sum(P) - 1.0, 1.0)

    @property
    def acf(self):
        """Normalized autocorrelation function, shape ``(max_lag+1,)``.

        Averaged over chains.  Returns ``None`` if no ACF data has been
        accumulated yet (``max_lag == 0``, ``decay != 1.0``, or no updates).

        Follows OnlineStats.jl AutoCov: per-lag autocovariance is recovered as

        .. math::

            C(k) = E[x_t x_{t-k}] - E_{\\rm lag}[x_{t-k}] \\cdot E_{\\rm cur}[x_t]

        using the actual pair-subset means for each lag (not the global mean).
        """
        if self.max_lag == 0:
            return None
        n = jnp.maximum(self._pair_count, 1.0)  # (n_chains, max_lag+1)
        cross = self._cross_sum / n  # E[x_t * x_{t-k}]
        m1 = self._m1_sum / n  # E[x_{t-k}]  (lagged mean)
        m2 = self._m2_sum / n  # E[x_t]      (current mean)
        cov_avg = jnp.mean(cross - m1 * m2, axis=0)  # average Cov over chains
        if cov_avg[0] <= 0:
            return None
        return cov_avg / cov_avg[0]

    @property
    def R_hat(self):
        """The split-R̂ convergence diagnostic.

        Compares intra-chain and inter-chain variance.  Returns ``NaN``
        when fewer than 2 chains have been observed.
        """
        if self.n_chains < 2:
            return _NaN
        chain_vars = self._chain_M2 / jnp.maximum(self._chain_count, 1.0)
        W = jnp.mean(jnp.real(chain_vars))
        if W <= 0:
            return _NaN
        N = jnp.mean(self._chain_count)
        B = jnp.var(jnp.real(self._chain_mean))
        return sqrt((N - 1) / N + B / W)

    @property
    def n_samples(self):
        """Total number of raw samples accumulated (never decayed)."""
        return self._n_samples_total

    # ------------------------------------------------------------------
    # Stats conversion
    # ------------------------------------------------------------------

    def get_stats(self) -> Stats:
        """Convert accumulated state into a :class:`Stats` object."""
        if jnp.sum(self._chain_count) == 0.0:
            return Stats()
        mean = self.mean
        variance = self.variance
        tau_corr = self.tau_corr
        return Stats(
            mean=mean,
            error_of_mean=self._compute_error_of_mean(variance),
            variance=variance,
            tau_corr=tau_corr,
            R_hat=self.R_hat,
        )

    # ------------------------------------------------------------------
    # Logging protocol
    # ------------------------------------------------------------------

    def to_dict(self):
        return self.get_stats().to_dict()

    def to_compound(self):
        return self.get_stats().to_compound()

    def __repr__(self):
        return repr(self.get_stats())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_error_of_mean(self, variance) -> float:
        if self.n_chains > 1:
            # SE of grand mean = std(chain_means) / sqrt(n_chains)
            return jnp.sqrt(jnp.var(jnp.real(self._chain_mean)) / self.n_chains)
        # Single chain: choose formula matching the tau convention.
        # tau_corr_acf uses the Sokal/full convention (tau_int = 1 for IID), so
        #   Var(mean) = variance * tau_int / N  →  error = sqrt(variance * tau / N)
        # tau_corr_batch uses the half convention (tau_batch = 0 for IID), so
        #   tau_int = 2*tau_batch + 1  →  error = sqrt(variance * (2*tau+1) / N)
        tau_acf = self.tau_corr_acf
        if not isnan(tau_acf):
            return jnp.sqrt(variance * tau_acf / self._n_samples_total)
        tau_batch = self.tau_corr_batch
        if not isnan(tau_batch):
            return jnp.sqrt(variance * (2 * tau_batch + 1) / self._n_samples_total)
        return _NaN
