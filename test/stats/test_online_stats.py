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

import numpy as np
import pytest

import netket as nk
from netket.stats import statistics, online_statistics
from netket._src.stats.online_stats import OnlineStats, expand_max_lag

from .. import common

pytestmark = common.skipif_distributed


rng = np.random.default_rng(42)


def make_data(n_chains=8, n_samples=100, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_chains, n_samples))


# --- Test 1: Mean/variance consistency with batch statistics() ---


def test_mean_variance_vs_batch():
    data = make_data(n_chains=8, n_samples=200)
    est = online_statistics(data)
    batch = statistics(data)

    np.testing.assert_allclose(est.mean, float(np.real(batch.mean)), rtol=1e-10)
    np.testing.assert_allclose(est.variance, float(batch.variance), rtol=1e-10)


# --- Test 2: Incremental consistency (chunked vs one-shot) ---


def test_incremental_consistency():
    data = make_data(n_chains=6, n_samples=300)
    # One-shot
    est_full = online_statistics(data)

    # Chunked into 10 pieces along samples axis
    chunks = np.split(data, 10, axis=1)
    est_chunked = None
    for chunk in chunks:
        est_chunked = online_statistics(chunk, est_chunked)

    np.testing.assert_allclose(est_full.mean, est_chunked.mean, rtol=1e-10)
    np.testing.assert_allclose(est_full.variance, est_chunked.variance, rtol=1e-10)


# --- Test: error_of_mean consistency with batch statistics() ---


def test_error_of_mean_vs_batch():
    """online_statistics and statistics agree on error_of_mean for multi-chain data.

    Both use sqrt(var(chain_means) / n_chains) so the result is exact.
    The old statistics() falls back to a blocking estimator when n_chains < 32,
    so we use 64 chains to ensure both implementations take the inter-chain path.
    """
    data = make_data(n_chains=64, n_samples=400)
    est = online_statistics(data)
    batch = statistics(data)

    np.testing.assert_allclose(
        est.get_stats().error_of_mean,
        float(batch.error_of_mean),
        rtol=1e-10,
        err_msg="error_of_mean must match between online_statistics and statistics",
    )


def test_error_of_mean_chunked_vs_oneshot():
    """online_statistics error_of_mean is identical whether data arrives in one
    shot or split into 5 chunks, and matches batch statistics().
    """
    # 64 chains: both online and batch statistics use the inter-chain formula
    data = make_data(n_chains=64, n_samples=500)

    # One-shot
    est_full = online_statistics(data)

    # 5-chunk
    chunks = np.split(data, 5, axis=1)
    est_chunked = None
    for chunk in chunks:
        est_chunked = online_statistics(chunk, est_chunked)

    # Chunked == one-shot (exact)
    np.testing.assert_allclose(
        est_full.get_stats().error_of_mean,
        est_chunked.get_stats().error_of_mean,
        rtol=1e-10,
        err_msg="error_of_mean must be identical for one-shot vs 5-chunk online update",
    )

    # Both == batch statistics (exact for inter-chain formula, n_chains >= 32)
    batch = statistics(data)
    np.testing.assert_allclose(
        est_full.get_stats().error_of_mean,
        float(batch.error_of_mean),
        rtol=1e-10,
        err_msg="error_of_mean must match between online_statistics and statistics",
    )


# --- Test 3: tau_corr (wide tolerance) ---


def test_tau_corr_reasonable():
    # iid data => tau_corr should be near 0 but >= 0
    data = make_data(n_chains=16, n_samples=500)
    est = online_statistics(data)
    assert est.tau_corr >= 0.0
    assert math.isfinite(est.tau_corr)


# --- Test 4: R_hat detects diverging chains ---


def test_R_hat_diverging():
    rng = np.random.default_rng(7)
    n_chains = 8
    # Half chains centered at 0, half at 10
    data = np.vstack(
        [
            rng.standard_normal((n_chains // 2, 100)),
            rng.standard_normal((n_chains // 2, 100)) + 10.0,
        ]
    )
    est = online_statistics(data)
    assert est.R_hat > 1.01, f"R_hat={est.R_hat} should be > 1.01 for diverging chains"


# --- Test 5: R_hat good for iid chains ---


def test_R_hat_good_chains():
    data = make_data(n_chains=16, n_samples=500)
    est = online_statistics(data)
    assert est.R_hat < 1.1, f"R_hat={est.R_hat} should be < 1.1 for iid chains"


# --- Test 6: Decay forgets old data ---


def test_decay_forgets_old():
    rng = np.random.default_rng(99)
    # First 50 updates: data near 0
    est = None
    for _ in range(50):
        batch = rng.standard_normal((4, 50))
        est = online_statistics(batch, est, decay=0.01)

    # Then 50 updates: data near 10
    for _ in range(50):
        batch = rng.standard_normal((4, 50)) + 10.0
        est = online_statistics(batch, est, decay=0.01)

    # Mean should be very close to 10
    assert (
        abs(est.mean - 10.0) < 0.5
    ), f"mean={est.mean} should be near 10 with strong decay"


# --- Test 7: tau_corr is finite with decay ---


def test_tau_corr_nan_with_decay():
    data = make_data(n_chains=8, n_samples=100)
    est = online_statistics(data, decay=0.99)
    assert math.isfinite(est.tau_corr), "tau_corr should be finite when decay < 1.0"


# --- Test 8: Complex data ---


def test_complex_data():
    rng = np.random.default_rng(13)
    data = rng.standard_normal((4, 100)) + 1j * rng.standard_normal((4, 100))
    est = online_statistics(data)
    expected_mean = np.mean(data)
    assert isinstance(est.mean, complex), "mean should be complex for complex data"
    np.testing.assert_allclose(est.mean, expected_mean, rtol=1e-10)


# --- Test 9: 1D single-chain input ---


def test_single_chain_1d():
    rng = np.random.default_rng(21)
    data = rng.standard_normal(200)
    est = online_statistics(data)
    np.testing.assert_allclose(est.mean, np.mean(data), rtol=1e-10)
    assert math.isnan(est.R_hat), "R_hat should be NaN for single chain"
    # With max_lag > 0 (default), tau_corr uses the ACF estimator and is valid
    # even for a single chain.
    assert math.isfinite(
        est.tau_corr
    ), "tau_corr should be finite via ACF for single chain"
    # Without ACF tracking, it falls back to between-chain which needs >= 2 chains.
    est_no_acf = online_statistics(data, max_lag=0)
    assert math.isnan(
        est_no_acf.tau_corr
    ), "tau_corr should be NaN for single chain without ACF"


# --- Test 14: ACF shape and normalization ---


def test_acf_shape_and_normalization():
    data = make_data(n_chains=4, n_samples=200)
    est = online_statistics(data, max_lag=16)
    acf = est.acf
    assert acf is not None, "acf should not be None when max_lag > 0"
    assert acf.shape == (17,), f"expected shape (17,), got {acf.shape}"
    np.testing.assert_allclose(acf[0], 1.0, rtol=1e-10, err_msg="acf[0] must be 1")


def test_tau_corr_acf_stable_across_chains():
    """tau_corr_acf should be stable (not vary wildly) as n_chains increases."""
    phi = 0.9
    taus = []
    for n_chains in [16, 64, 128, 256]:
        rng = np.random.default_rng(42)
        data = np.zeros((n_chains, 1000))
        data[:, 0] = rng.standard_normal(n_chains)
        for t in range(1, 1000):
            data[:, t] = phi * data[:, t - 1] + np.sqrt(
                1 - phi**2
            ) * rng.standard_normal(n_chains)
        est = online_statistics(data)
        taus.append(est.tau_corr_acf)

    # All estimates should be within 50% of each other (Sokal would vary by 2x+)
    taus = np.array(taus)
    assert (
        np.max(taus) / np.min(taus) < 1.5
    ), f"tau_corr_acf varies too much with n_chains: {taus}"


# --- Test 15: ACF is None when max_lag=0; with decay it is still computed ---


def test_acf_none_cases():
    data = make_data(n_chains=4, n_samples=100)
    assert online_statistics(data, max_lag=0).acf is None
    # With decay, the ACF is still computed (EMA-weighted); it is not None.
    assert online_statistics(data, decay=0.9).acf is not None


# --- Test 16: tau_corr_acf consistent with ACF ---


def test_tau_corr_acf_noniid():
    # AR(1) process with strong autocorrelation
    rng = np.random.default_rng(42)
    phi = 0.8
    n_chains, n_samples = 8, 512
    data = np.zeros((n_chains, n_samples))
    data[:, 0] = rng.standard_normal(n_chains)
    for t in range(1, n_samples):
        data[:, t] = phi * data[:, t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(
            n_chains
        )

    est = online_statistics(data, max_lag=64)
    # Theoretical tau = 1 + 2*phi/(1-phi) ≈ 1 + 2*4 = 9  (very roughly)
    assert (
        est.tau_corr_acf > 2.0
    ), f"tau_corr_acf={est.tau_corr_acf} should be > 2 for AR(1) phi=0.8"
    assert math.isfinite(est.tau_corr_acf)


# --- Test 17: ACF chunked vs one-shot consistency ---


def test_acf_chunked_vs_oneshot():
    data = make_data(n_chains=4, n_samples=300)
    est_full = online_statistics(data, max_lag=32)

    chunks = np.split(data, 10, axis=1)
    est_chunked = None
    for chunk in chunks:
        est_chunked = online_statistics(chunk, est_chunked, max_lag=32)

    np.testing.assert_allclose(
        est_full.acf,
        est_chunked.acf,
        atol=1e-12,
        err_msg="ACF should be identical for one-shot vs chunked update",
    )


# --- Test: n_chains and n_samples properties ---


def test_n_chains_and_n_samples():
    data = make_data(n_chains=6, n_samples=80)
    est = online_statistics(data)
    assert est.n_chains == 6
    assert est.n_samples == 6 * 80

    # n_samples accumulates across updates
    est2 = est.update(make_data(n_chains=6, n_samples=40))
    assert est2.n_samples == 6 * 80 + 6 * 40
    assert est2.n_chains == 6  # unchanged


# --- Test: tau_corr_batch ---


def test_tau_corr_batch():
    # With multiple chains, tau_corr_batch should be finite and >= 0
    data = make_data(n_chains=16, n_samples=500)
    est = online_statistics(data)
    assert math.isfinite(est.tau_corr_batch)
    assert est.tau_corr_batch >= 0.0

    # Single chain → NaN
    est_s = online_statistics(make_data(n_chains=1, n_samples=500))
    assert math.isnan(est_s.tau_corr_batch)

    # decay != 1 → tau_corr_batch is still computable (EMA-weighted between-chain estimate)
    est_d = online_statistics(data, decay=0.99)
    assert math.isfinite(est_d.tau_corr_batch)
    assert est_d.tau_corr_batch >= 0.0

    # With max_lag=0, tau_corr should fall back to tau_corr_batch for multi-chain
    est_no_acf = online_statistics(data, max_lag=0)
    assert math.isfinite(est_no_acf.tau_corr)
    assert est_no_acf.tau_corr == est_no_acf.tau_corr_batch


# --- Test: dtype preservation ---


def test_dtype_preservation():
    rng = np.random.default_rng(7)

    # float32 input → float32 mean
    data_f32 = rng.standard_normal((4, 100)).astype(np.float32)
    est = OnlineStats.from_data(data_f32)
    assert est._chain_mean.dtype == np.float32

    # float64 input → float64 mean
    data_f64 = rng.standard_normal((4, 100))
    est64 = OnlineStats.from_data(data_f64)
    assert est64._chain_mean.dtype == np.float64

    # complex128 input → complex128 mean, and mean is complex
    data_c = rng.standard_normal((4, 100)) + 1j * rng.standard_normal((4, 100))
    est_c = OnlineStats.from_data(data_c)
    assert est_c._chain_mean.dtype == np.complex128


# --- Test: tau_corr prefers ACF over batch ---


def test_tau_corr_prefers_acf():
    # With max_lag > 0, tau_corr == tau_corr_acf (not tau_corr_batch)
    data = make_data(n_chains=8, n_samples=200)
    est = online_statistics(data, max_lag=32)
    assert est.tau_corr == est.tau_corr_acf

    # With max_lag=0, tau_corr falls back to tau_corr_batch
    est_no_acf = online_statistics(data, max_lag=0)
    assert est_no_acf.tau_corr == est_no_acf.tau_corr_batch


# --- Test 10: Logging protocol ---


def test_logging_protocol():
    data = make_data(n_chains=4, n_samples=100)
    est = online_statistics(data)

    # repr works
    r = repr(est)
    assert isinstance(r, str) and len(r) > 0

    # to_dict works
    d = est.to_dict()
    assert "Mean" in d
    assert "Variance" in d
    assert "Sigma" in d
    assert "R_hat" in d
    assert "TauCorr" in d

    # to_compound works
    key, compound = est.to_compound()
    assert key == "Mean"
    assert compound == d

    # get_stats returns a Stats object
    stats = est.get_stats()
    assert isinstance(stats, nk.stats.Stats)


# --- Test 11: Mismatched n_chains raises ValueError ---


def test_mismatched_chains_raises():
    data1 = make_data(n_chains=4, n_samples=100)
    data2 = make_data(n_chains=8, n_samples=100)
    est = online_statistics(data1)
    with pytest.raises(ValueError, match="Number of chains"):
        est.update(data2)


# --- Test 12: Zero-count state returns empty Stats ---


def test_empty_state_get_stats():
    # There is no "empty" constructor; get_stats() on a zero-count accumulator
    # (all chain counts decayed to zero) should return Stats with NaN fields.
    # The simplest proxy: check that Stats() itself has NaN fields.
    from netket.stats import Stats

    stats = Stats()
    assert math.isnan(stats.mean)
    assert math.isnan(stats.variance)


# --- Test 13: Immutability of update ---


def test_update_immutability():
    data = make_data(n_chains=4, n_samples=50)
    est1 = online_statistics(data)
    old_count = float(np.sum(est1._chain_count))

    data2 = make_data(n_chains=4, n_samples=50, seed=99)
    est2 = est1.update(data2)

    # est1 should be unchanged
    assert float(np.sum(est1._chain_count)) == old_count
    assert float(np.sum(est2._chain_count)) > old_count


# --- Tests for expand_max_lag ---


def test_expand_max_lag_shapes():
    """Buffer shapes update correctly after expansion."""
    n_chains, old_lag, new_lag = 4, 16, 48
    data = make_data(n_chains=n_chains, n_samples=200)
    est = online_statistics(data, max_lag=old_lag)
    expanded = expand_max_lag(est, new_lag)

    assert expanded.max_lag == new_lag
    assert expanded._cross_sum.shape == (n_chains, new_lag + 1)
    assert expanded._m1_sum.shape == (n_chains, new_lag + 1)
    assert expanded._m2_sum.shape == (n_chains, new_lag + 1)
    assert expanded._pair_count.shape == (n_chains, new_lag + 1)
    assert expanded._chain_buf.shape == (n_chains, new_lag)


def test_expand_max_lag_old_lags_preserved():
    """ACF data for lags 0..old_max_lag is identical after expansion."""
    data = make_data(n_chains=4, n_samples=300)
    est = online_statistics(data, max_lag=16)
    expanded = expand_max_lag(est, 64)

    np.testing.assert_array_equal(
        est._cross_sum,
        expanded._cross_sum[:, :17],
        err_msg="cross_sum for old lags must be unchanged",
    )
    np.testing.assert_array_equal(
        est._pair_count,
        expanded._pair_count[:, :17],
        err_msg="pair_count for old lags must be unchanged",
    )
    np.testing.assert_array_equal(
        est.acf, expanded.acf[:17], err_msg="ACF values for old lags must be identical"
    )


def test_expand_max_lag_new_lags_zero():
    """New lag columns start with zero accumulators."""
    data = make_data(n_chains=4, n_samples=200)
    old_lag = 16
    est = online_statistics(data, max_lag=old_lag)
    expanded = expand_max_lag(est, 48)

    np.testing.assert_array_equal(
        expanded._cross_sum[:, old_lag + 1 :],
        0,
        err_msg="new lag cross_sum columns must be zero",
    )
    np.testing.assert_array_equal(
        expanded._pair_count[:, old_lag + 1 :],
        0,
        err_msg="new lag pair_count columns must be zero",
    )


def test_expand_max_lag_buffer_right_aligned():
    """The rolling buffer valid region stays right-aligned after expansion."""
    n_chains, old_lag, new_lag = 4, 16, 32
    data = make_data(n_chains=n_chains, n_samples=200)
    est = online_statistics(data, max_lag=old_lag)
    expanded = expand_max_lag(est, new_lag)

    # The last old_lag columns of the new buffer must equal the old buffer.
    np.testing.assert_array_equal(
        est._chain_buf,
        expanded._chain_buf[:, new_lag - old_lag :],
        err_msg="existing buffer samples must be right-aligned in expanded buffer",
    )
    # The prepended columns must be zero.
    np.testing.assert_array_equal(
        expanded._chain_buf[:, : new_lag - old_lag],
        0,
        err_msg="prepended buffer columns must be zero",
    )


def test_expand_max_lag_welford_unchanged():
    """Mean, variance, n_samples, and n_chains are unaffected by expansion."""
    data = make_data(n_chains=6, n_samples=150)
    est = online_statistics(data, max_lag=16)
    expanded = expand_max_lag(est, 64)

    np.testing.assert_allclose(est.mean, expanded.mean, rtol=1e-14)
    np.testing.assert_allclose(est.variance, expanded.variance, rtol=1e-14)
    assert est.n_samples == expanded.n_samples
    assert est.n_chains == expanded.n_chains


def test_expand_max_lag_subsequent_updates():
    """After expansion, further updates work and ACF extends to the new lags."""
    rng_loc = np.random.default_rng(77)
    n_chains = 4

    # Accumulate some data with a small window, then expand.
    batch1 = rng_loc.standard_normal((n_chains, 100))
    est = online_statistics(batch1, max_lag=16)
    est = expand_max_lag(est, 64)

    # Feed more data — should not raise and ACF should be valid.
    for _ in range(5):
        batch = rng_loc.standard_normal((n_chains, 50))
        est = est.update(batch)

    assert est.max_lag == 64
    acf = est.acf
    assert acf is not None
    assert acf.shape == (65,)
    np.testing.assert_allclose(acf[0], 1.0, rtol=1e-10)

    # New lags (>16) should now have non-zero pair counts from the extra batches.
    assert (
        float(est._pair_count[:, 20:].sum()) > 0
    ), "new lags should accumulate pairs after subsequent updates"


def test_expand_max_lag_convergence():
    """With enough data after expansion, tau_corr_acf converges to the one-shot value."""
    rng_loc = np.random.default_rng(55)
    phi = 0.7
    n_chains, n_samples = 8, 1000

    data = np.zeros((n_chains, n_samples))
    data[:, 0] = rng_loc.standard_normal(n_chains)
    for t in range(1, n_samples):
        data[:, t] = phi * data[:, t - 1] + np.sqrt(
            1 - phi**2
        ) * rng_loc.standard_normal(n_chains)

    # Reference: one-shot with large window.
    est_ref = online_statistics(data, max_lag=64)
    tau_ref = est_ref.tau_corr_acf

    # Split: first half with small window, expand, then second half.
    half = n_samples // 2
    est = online_statistics(data[:, :half], max_lag=8)
    est = expand_max_lag(est, 64)
    est = est.update(data[:, half:])

    # The estimate won't be identical (new lags miss the first half) but
    # should be in the same ballpark (within 50%).
    assert math.isfinite(est.tau_corr_acf)
    assert (
        abs(est.tau_corr_acf - tau_ref) / tau_ref < 0.5
    ), f"tau after expand ({est.tau_corr_acf:.2f}) too far from reference ({tau_ref:.2f})"


def test_expand_max_lag_from_zero():
    """Expanding from max_lag=0 (ACF disabled) allocates fresh buffers."""
    data = make_data(n_chains=4, n_samples=100)
    est = online_statistics(data, max_lag=0)
    assert est.acf is None

    expanded = expand_max_lag(est, 32)
    assert expanded.max_lag == 32
    assert expanded._cross_sum.shape == (4, 33)
    assert expanded._chain_buf.shape == (4, 32)

    # After feeding more data, ACF should be valid.
    extra = make_data(n_chains=4, n_samples=100)
    expanded = expanded.update(extra)
    assert expanded.acf is not None


def test_rolling_buffer_large_batch():
    """n_batch >= max_lag branch: buffer becomes x[:, -max_lag:] and ACF stays valid."""
    max_lag = 8
    rng_loc = np.random.default_rng(17)
    big_batch = rng_loc.standard_normal((4, 12))  # n_batch=12 > max_lag=8

    est = online_statistics(big_batch, max_lag=max_lag)

    np.testing.assert_array_equal(np.asarray(est._chain_buf), big_batch[:, -max_lag:])
    assert int(est._buf_len) == max_lag

    # Further small updates keep ACF valid.
    for _ in range(5):
        est = est.update(rng_loc.standard_normal((4, 6)))
    assert est.acf is not None and math.isfinite(est.tau_corr)


def test_expand_max_lag_invalid():
    """expand_max_lag raises ValueError when new_max_lag is not strictly larger."""
    data = make_data(n_chains=4, n_samples=100)
    est = online_statistics(data, max_lag=32)

    with pytest.raises(ValueError, match="must be >"):
        expand_max_lag(est, 32)  # same value

    with pytest.raises(ValueError, match="must be >"):
        expand_max_lag(est, 16)  # smaller value


# --- Tests for max_lag=0 (ACF disabled) ---


def test_max_lag_zero_basic():
    """max_lag=0: ACF disabled, mean/variance still correct, repr works."""
    data = make_data(n_chains=4, n_samples=200)
    est = online_statistics(data, max_lag=0)

    assert est.acf is None and math.isnan(est.tau_corr_acf)
    assert est._chain_buf.shape == (4, 0)
    np.testing.assert_allclose(est.mean, np.mean(data), rtol=1e-10)
    np.testing.assert_allclose(est.variance, np.var(data), rtol=1e-10)
    assert isinstance(repr(est), str)


def test_max_lag_zero_incremental():
    """Chunked updates with max_lag=0 give identical mean/variance to one-shot."""
    data = make_data(n_chains=6, n_samples=300)
    chunks = np.split(data, 10, axis=1)
    est = None
    for chunk in chunks:
        est = online_statistics(chunk, est, max_lag=0)
    np.testing.assert_allclose(est.mean, np.mean(data), rtol=1e-10)
    np.testing.assert_allclose(est.variance, np.var(data), rtol=1e-10)
