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

"""Low-level JIT-compiled kernels for the online ACF + parallel Welford update."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


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
        old_buf_len: Number of valid samples in ``chain_buf`` (JAX integer,
            dynamic — used only for masked cross-batch pair counting).

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
        # Shift the right-aligned buffer left by n_batch and append new samples.
        # chain_buf has shape (n_chains, max_lag) with valid data right-aligned;
        # this preserves that invariant without needing old_buf_len for shapes.
        new_buf = jnp.concatenate([chain_buf[:, n_batch:], x], axis=1)

    return cross_sum, m1_sum, m2_sum, pair_count, new_buf


@partial(jax.jit, static_argnames=("max_lag",))
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

    ``decay`` and ``max_lag`` are static arguments so that
    (a) the ``if decay != 1.0`` branch is resolved at compile time and (b) the
    scan length ``max_lag + 1`` is a compile-time constant.
    ``old_buf_len`` is a dynamic JAX integer (no longer static) so that
    re-filling the rolling buffer does not trigger recompilation.

    Returns:
        ``(chain_count, chain_mean, chain_M2,
           cross_sum, m1_sum, m2_sum, pair_count, chain_buf)``
        — updated JAX arrays only.  Python-level counters (``_n_samples_total``,
        ``_buf_len``) are computed by the caller.
    """
    n_chains, n_samples_per_chain = data.shape

    # ---- Parallel Welford merge ----------------------------------------
    mean_dtype = chain_mean.dtype  # preserve input dtype (e.g. float32, complex128)
    c_count = n_samples_per_chain
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
    ratio = c_count / safe_count
    chain_mean = (chain_mean + delta * ratio).astype(mean_dtype)
    chain_M2 = (
        chain_M2 + c_M2 + jnp.abs(delta) ** 2 * (chain_count * c_count / safe_count)
    )
    chain_count = count_new

    # ---- Online ACF update (real part only) ----------------------------
    if max_lag > 0:
        if decay is not None:
            cross_sum = cross_sum * decay
            m1_sum = m1_sum * decay
            m2_sum = m2_sum * decay
            pair_count = pair_count * decay

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
