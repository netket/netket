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

"""Online statistics accumulator for streaming MCMC data."""

from functools import partial
from math import isnan, sqrt

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from netket.utils import struct

from netket.stats.mc_stats import Stats

_NaN = float("nan")


# ------------------------------------------------------------------
# Pure helpers (module-level, JIT-compiled)
# ------------------------------------------------------------------


def _acf_core(
    x, max_lag, cross_sum, m1_sum, m2_sum, pair_count, chain_buf, old_buf_len
):
    """Vectorized ACF accumulator update via ``lax.scan`` (called inside JIT).

    Replaces the Python ``for k in range(max_lag + 1)`` loop with a single
    ``lax.scan`` over all lags.  All array shapes are static; dynamic start
    indices are handled with ``lax.dynamic_slice_in_dim``.

    Args:
        x: Real part of data, shape ``(n_chains, n_batch)``.
        max_lag: Maximum lag (static Python int).
        cross_sum, m1_sum, m2_sum, pair_count: ACF accumulators,
            shape ``(n_chains, max_lag + 1)``.
        chain_buf: Rolling sample buffer, shape ``(n_chains, max_lag)``.
        old_buf_len: Number of valid samples in ``chain_buf`` (static Python int).

    Returns:
        Updated ``(cross_sum, m1_sum, m2_sum, pair_count, new_buf)``.
    """
    n_chains, n_batch = x.shape

    # Padded views for static-size window extraction.
    # x_lpad: max_lag zeros on the left → lag_shift[j] = x[j-k] for j>=k, else 0.
    x_lpad = jnp.pad(x, ((0, 0), (max_lag, 0)))  # (n_chains, max_lag + n_batch)
    # x_rpad: max_lag zeros on the right → safe cross-batch cur reads.
    x_rpad = jnp.pad(x, ((0, 0), (0, max_lag)))  # (n_chains, n_batch + max_lag)
    # buf_rpad: max_lag zeros on the right → safe cross-batch lag reads.
    buf_rpad = jnp.pad(chain_buf, ((0, 0), (0, max_lag)))  # (n_chains, 2 * max_lag)

    def scan_body(carry, k):
        cross, m1, m2, pc = carry

        # --- Within-batch pairs at lag k ---
        # lag_w[j] = x[j-k] for j >= k (left-padded zeros fill j < k).
        lag_w = lax.dynamic_slice_in_dim(x_lpad, max_lag - k, n_batch, axis=1)
        valid_w = jnp.arange(n_batch) >= k  # (n_batch,)
        cur_w = jnp.where(valid_w[None, :], x, 0.0)
        d_cross_w = jnp.sum(cur_w * lag_w, axis=1)  # (n_chains,)
        d_m1_w = jnp.sum(lag_w, axis=1)  # lag_w already 0 for invalid j
        d_m2_w = jnp.sum(cur_w, axis=1)
        d_n_w = jnp.sum(valid_w).astype(jnp.float64)

        # --- Cross-batch pairs at lag k (current batch × rolling buffer) ---
        j_lo = jnp.maximum(0, k - old_buf_len)
        j_hi = jnp.minimum(k, n_batch) - 1
        n_cross = jnp.maximum(0, j_hi - j_lo + 1)
        buf_start = max_lag - jnp.minimum(k, old_buf_len)

        # Extract max_lag-wide windows (out-of-range regions are zero-padded
        # or JAX-clamped, then masked out by valid_c).
        valid_c = jnp.arange(max_lag) < n_cross  # (max_lag,)
        cur_c_raw = lax.dynamic_slice_in_dim(x_rpad, j_lo, max_lag, axis=1)
        lag_c_raw = lax.dynamic_slice_in_dim(buf_rpad, buf_start, max_lag, axis=1)
        cur_c = jnp.where(valid_c[None, :], cur_c_raw, 0.0)
        lag_c = jnp.where(valid_c[None, :], lag_c_raw, 0.0)
        d_cross_c = jnp.sum(cur_c * lag_c, axis=1)
        d_m1_c = jnp.sum(lag_c, axis=1)
        d_m2_c = jnp.sum(cur_c, axis=1)
        d_n_c = n_cross.astype(jnp.float64)

        # Scatter-add both contributions into lag-k column.
        new_cross = cross.at[:, k].add(d_cross_w + d_cross_c)
        new_m1 = m1.at[:, k].add(d_m1_w + d_m1_c)
        new_m2 = m2.at[:, k].add(d_m2_w + d_m2_c)
        new_pc = pc.at[:, k].add(d_n_w + d_n_c)

        return (new_cross, new_m1, new_m2, new_pc), None

    (cross_sum, m1_sum, m2_sum, pair_count), _ = lax.scan(
        scan_body,
        (cross_sum, m1_sum, m2_sum, pair_count),
        jnp.arange(max_lag + 1),
    )

    # Rolling buffer: keep the last max_lag real samples per chain.
    # All branch conditions are Python-static (n_batch and max_lag are compile-
    # time constants), so only one branch is traced per JIT specialisation.
    if n_batch >= max_lag:
        new_buf = x[:, -max_lag:]
    else:
        # Append new samples to the valid portion of the old buffer.
        old_valid = chain_buf[:, max_lag - old_buf_len :]  # (n_chains, old_buf_len)
        combined = jnp.concatenate([old_valid, x], axis=1)
        keep = min(combined.shape[1], max_lag)
        new_buf = (
            jnp.zeros((n_chains, max_lag), dtype=jnp.float64)
            .at[:, max_lag - keep :]
            .set(combined[:, -keep:])
        )

    return cross_sum, m1_sum, m2_sum, pair_count, new_buf


@partial(jax.jit, static_argnames=("max_lag", "old_buf_len"))
def _update_arrays(
    chain_count,
    chain_mean,
    chain_M2,
    cross_sum,
    m1_sum,
    m2_sum,
    pair_count,
    chain_buf,
    decay,
    max_lag,
    old_buf_len,
    data,
):
    """JIT-compiled core: parallel Welford merge + online ACF update.

    ``decay``, ``max_lag``, and ``old_buf_len`` are static arguments so that
    (a) the ``if decay != 1.0`` branch is resolved at compile time, (b) the
    scan length ``max_lag + 1`` is a compile-time constant, and (c) the
    cross-batch slice offsets derived from ``old_buf_len`` are constants.

    Returns:
        ``(chain_count, chain_mean, chain_M2,
           cross_sum, m1_sum, m2_sum, pair_count, chain_buf)``
        — updated JAX arrays only.  Python-level counters (``_n_samples_total``,
        ``_buf_len``) are computed by the caller.
    """
    n_chains, n_samples_per_chain = data.shape
    print("compiling with 'old_buf_len'", old_buf_len)

    # ---- Parallel Welford merge ----------------------------------------
    c_count = float(n_samples_per_chain)
    c_mean = jnp.mean(data, axis=1)  # (n_chains,)
    c_M2 = jnp.sum(
        jnp.abs(data - c_mean[:, None]) ** 2, axis=1
    ).real  # (n_chains,) real

    if decay is not None:
        chain_count = chain_count * decay
        chain_M2 = chain_M2 * decay

    count_new = chain_count + c_count
    # Guard against zero denominator on the very first decayed call.
    safe_count = jnp.where(count_new > 0, count_new, 1.0)
    delta = c_mean - chain_mean
    # Cast the update ratio to chain_mean's dtype to avoid float32→float64 promotion.
    ratio = (c_count / safe_count).astype(chain_mean.dtype)
    chain_mean = chain_mean + delta * ratio
    chain_M2 = (
        chain_M2 + c_M2 + jnp.abs(delta) ** 2 * (chain_count * c_count / safe_count)
    )
    chain_count = count_new

    # ---- Online ACF update (real part only) ----------------------------
    if max_lag > 0:
        x = jnp.real(data)
        cross_sum, m1_sum, m2_sum, pair_count, chain_buf = _acf_core(
            x, max_lag, cross_sum, m1_sum, m2_sum, pair_count, chain_buf, old_buf_len
        )

    return (
        chain_count,
        chain_mean,
        chain_M2,
        cross_sum,
        m1_sum,
        m2_sum,
        pair_count,
        chain_buf,
    )


# ------------------------------------------------------------------
# OnlineStats dataclass
# ------------------------------------------------------------------


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
    _decay: float | None
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
    _buf_len: int = struct.field(pytree_node=False, default=0)

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

        # Python-side counters (not traced by JAX).
        new_buf_len = min(self._buf_len + n_samples_per_chain, self.max_lag)

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
        total = float(jnp.sum(self._chain_count))
        if total == 0.0:
            return _NaN
        m = (jnp.sum(self._chain_count * self._chain_mean) / total).item()
        if isinstance(m, complex) and m.imag == 0.0:
            return m.real
        return m

    @property
    def variance(self):
        """The current variance estimate (float64)."""
        total = float(jnp.sum(self._chain_count))
        if total == 0.0:
            return _NaN
        global_mean = jnp.sum(self._chain_count * self._chain_mean) / total
        global_M2 = float(
            jnp.sum(self._chain_M2)
            + jnp.sum(self._chain_count * jnp.abs(self._chain_mean - global_mean) ** 2)
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
        if self._decay is not None or self.n_chains < 2:
            return _NaN
        variance = self.variance
        if isnan(variance) or variance <= 0:
            return _NaN
        n_per_chain_eff = self._n_samples_total / self.n_chains
        B = float(jnp.var(jnp.real(self._chain_mean)))
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
        return float(max(2.0 * np.sum(P) - 1.0, 1.0))

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
        if self.max_lag == 0 or self._decay is not None:
            return None
        n = jnp.maximum(self._pair_count, 1.0)  # (n_chains, max_lag+1)
        cross = self._cross_sum / n  # E[x_t * x_{t-k}]
        m1 = self._m1_sum / n  # E[x_{t-k}]  (lagged mean)
        m2 = self._m2_sum / n  # E[x_t]      (current mean)
        cov_avg = jnp.mean(cross - m1 * m2, axis=0)  # average Cov over chains
        if float(cov_avg[0]) <= 0:
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
        W = float(jnp.mean(jnp.real(chain_vars)))
        if W <= 0:
            return _NaN
        N = float(jnp.mean(self._chain_count))
        B = float(jnp.var(jnp.real(self._chain_mean)))
        return float(sqrt((N - 1) / N + B / W))

    @property
    def n_samples(self):
        """Total number of raw samples accumulated (never decayed)."""
        return self._n_samples_total

    # ------------------------------------------------------------------
    # Stats conversion
    # ------------------------------------------------------------------

    def get_stats(self) -> Stats:
        """Convert accumulated state into a :class:`Stats` object."""
        if float(jnp.sum(self._chain_count)) == 0.0:
            return Stats()
        mean = self.mean
        variance = self.variance
        tau_corr = self.tau_corr
        return Stats(
            mean=mean,
            error_of_mean=self._compute_error_of_mean(variance, tau_corr),
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

    def _compute_error_of_mean(self, variance, tau_corr) -> float:
        if self.n_chains > 1:
            return float(
                sqrt(float(jnp.var(jnp.real(self._chain_mean))) / self.n_chains)
            )
        elif not isnan(tau_corr):
            return float(sqrt(variance * (2 * tau_corr + 1) / self._n_samples_total))
        return _NaN


# ------------------------------------------------------------------
# Functional API
# ------------------------------------------------------------------


@partial(jax.jit, static_argnames=("max_lag"))
def online_statistics(
    data, old_estimator=None, *, decay: float | None = None, max_lag: int = 64
) -> OnlineStats:
    """Accumulate streaming MCMC statistics across batches.

    This is the functional API for :class:`OnlineStats`.  Each call merges
    a new batch of data into the running estimator using the parallel Welford
    algorithm.

    Args:
        data: Array of shape ``(n_samples,)`` or ``(n_chains, n_samples_per_chain)``.
        old_estimator: Previous :class:`OnlineStats` instance, or ``None`` to start fresh.
        decay: EMA decay factor applied per call (default 1.0 = no decay).
            When ``decay < 1.0``, old data is down-weighted exponentially and
            ``tau_corr`` is set to ``NaN``.
        max_lag: Maximum lag for the online ACF estimator (default 64).
            Set to 0 to disable ACF tracking.

    Returns:
        Updated :class:`OnlineStats` instance.

    Example::

        estimator = None
        for batch in training_loop:
            estimator = nk.stats.online_statistics(batch, estimator)
        stats = estimator.get_stats()
    """
    if decay is not None:
        decay = jnp.float(decay)
    print("i am compiling something")
    if old_estimator is None:
        old_estimator = OnlineStats(
            data.shape[0], dtype=data.dtype, decay=decay, max_lag=max_lag
        )

    return old_estimator.update(data)


def expand_max_lag(estimator: OnlineStats, new_max_lag: int) -> OnlineStats:
    """Return a copy of *estimator* with ``max_lag`` expanded to ``new_max_lag``.

    **Rigorousness**: Each lag's ACF estimator is normalised by its own
    ``pair_count``, so it is always unbiased.  Lags ``0..old_max_lag`` are
    preserved exactly — their accumulators are untouched.  New lags
    ``old_max_lag+1..new_max_lag`` start with zero pairs and accumulate from
    the next :meth:`~OnlineStats.update` call onward.  The only loss of
    information is that historical samples (beyond the rolling buffer) cannot
    be retroactively used for the new lags.

    The rolling buffer is right-aligned by convention (see ``_acf_core``):
    left-padding it with zeros preserves the alignment, so the existing
    ``_buf_len`` samples will immediately contribute cross-batch pairs at the
    new lags on the very next update.

    The Welford mean/variance state is completely unaffected.

    Args:
        estimator: Existing :class:`OnlineStats` instance.
        new_max_lag: Target maximum lag; must be strictly greater than
            ``estimator.max_lag``.

    Returns:
        New :class:`OnlineStats` with expanded ACF buffers.

    Raises:
        ValueError: if ``new_max_lag <= estimator.max_lag``.
    """
    old_max_lag = estimator.max_lag
    new_max_lag = int(new_max_lag)
    if new_max_lag <= old_max_lag:
        raise ValueError(
            f"new_max_lag={new_max_lag} must be > current max_lag={old_max_lag}"
        )

    old_acf_len = old_max_lag + 1 if old_max_lag > 0 else 0
    new_acf_len = new_max_lag + 1
    extra_acf = new_acf_len - old_acf_len  # extra columns for new lags

    # Append zeros for new lags (axis=1).  Works for old_max_lag==0 too,
    # where old ACF arrays have shape (n_chains, 0).
    new_cross_sum = jnp.pad(estimator._cross_sum, ((0, 0), (0, extra_acf)))
    new_m1_sum = jnp.pad(estimator._m1_sum, ((0, 0), (0, extra_acf)))
    new_m2_sum = jnp.pad(estimator._m2_sum, ((0, 0), (0, extra_acf)))
    new_pair_count = jnp.pad(estimator._pair_count, ((0, 0), (0, extra_acf)))

    # Expand the rolling buffer: prepend zeros on the LEFT so valid samples
    # remain right-aligned (the layout expected by _acf_core).
    extra_buf = new_max_lag - old_max_lag
    new_chain_buf = jnp.pad(estimator._chain_buf, ((0, 0), (extra_buf, 0)))

    return estimator.replace(
        max_lag=new_max_lag,
        _cross_sum=new_cross_sum,
        _m1_sum=new_m1_sum,
        _m2_sum=new_m2_sum,
        _pair_count=new_pair_count,
        _chain_buf=new_chain_buf,
    )


def thin_acf_by_2(estimator: OnlineStats) -> OnlineStats:
    """Re-index ACF accumulators for a 2× coarser sample rate.

    Call this immediately *before* doubling the MCMC sweep size so that the
    accumulated ACF data stays in sync with the new sampling rate.

    **Why this is necessary**: each ACF accumulator column stores pairs at a
    physical lag of ``k × sweep_size``.  If you double ``sweep_size`` without
    adjusting the accumulators, future pairs at lag ``k`` (physical lag
    ``k × 2 × old_sweep``) are mixed with old pairs at lag ``k`` (physical
    lag ``k × old_sweep``) — two incompatible scales in the same column.

    **What this does**: keeps only the even-indexed lag columns (``0, 2, 4,
    …``) and relabels them as lags ``0, 1, 2, …``.  This is rigorous because
    old lag ``2k`` already estimates ``C(2k × old_sweep) = C(k × new_sweep)``,
    which is exactly what new lag ``k`` should estimate.  Odd-lag columns are
    discarded — they have no counterpart in the new 2× grid.

    The result has ``max_lag = old_max_lag // 2``.  Chain this with
    :func:`expand_max_lag` to restore the original window width::

        stats = thin_acf_by_2(stats)
        stats = expand_max_lag(stats, old_max_lag)

    The rolling buffer is subsampled by the same factor: every other stored
    sample is kept so that cross-batch pair computation at the new rate is
    approximately correct.  The Welford mean/variance state is unaffected.

    Args:
        estimator: Existing :class:`OnlineStats` instance with ``max_lag >= 2``.

    Returns:
        New :class:`OnlineStats` with ``max_lag = old_max_lag // 2`` and
        re-indexed ACF accumulators.

    Raises:
        ValueError: if ``max_lag < 2``.
    """
    old_max_lag = estimator.max_lag
    if old_max_lag < 2:
        raise ValueError(f"max_lag={old_max_lag} must be >= 2 to thin by 2")

    new_max_lag = old_max_lag // 2

    # Keep even-indexed lag columns: 0, 2, 4, …, 2*new_max_lag.
    # Static slice: step=2, fixed count new_max_lag+1 — no dynamic shapes.
    even_cols = (
        jnp.arange(new_max_lag + 1) * 2
    )  # (new_max_lag+1,) compile-time constant
    new_cross_sum = estimator._cross_sum[:, even_cols]
    new_m1_sum = estimator._m1_sum[:, even_cols]
    new_m2_sum = estimator._m2_sum[:, even_cols]
    new_pair_count = estimator._pair_count[:, even_cols]

    # Subsample the rolling buffer: keep every other sample, starting from
    # index -(2*new_max_lag) so we always get exactly new_max_lag elements
    # with the rightmost (most recent) slice consistent across even/odd old_max_lag.
    new_chain_buf = estimator._chain_buf[
        :, -(2 * new_max_lag) :: 2
    ]  # (n_chains, new_max_lag)

    return estimator.replace(
        max_lag=new_max_lag,
        _cross_sum=new_cross_sum,
        _m1_sum=new_m1_sum,
        _m2_sum=new_m2_sum,
        _pair_count=new_pair_count,
        _chain_buf=new_chain_buf,
        _buf_len=estimator._buf_len // 2,
    )
