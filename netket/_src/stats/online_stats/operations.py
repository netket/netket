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

"""Functional API and ACF-window operations for :class:`OnlineStats`."""

from functools import partial

import jax
import jax.numpy as jnp

from netket._src.stats.online_stats.accumulator import OnlineStats


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
    if old_estimator is None:
        # Determine n_chains after the same reshape that update() will apply.
        n_chains = 1 if data.ndim == 1 else data.shape[0]
        old_estimator = OnlineStats(
            n_chains, dtype=data.dtype, decay=decay, max_lag=max_lag
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
