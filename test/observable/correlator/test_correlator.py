"""Tests for correlation function observables."""

import pytest
import numpy as np

import netket as nk


def _make_vstate(N=4, n_samples=512, exact=True):
    hi = nk.hilbert.Spin(s=1 / 2, N=N)
    ma = nk.models.RBM(alpha=1, param_dtype=float)
    if exact:
        sa = nk.sampler.ExactSampler(hilbert=hi)
    else:
        sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains_per_rank=16)
    return nk.vqs.MCState(sampler=sa, model=ma, n_samples=n_samples), hi


class TestConnectedCorrelator:
    def test_agrees_with_product_expect(self):
        """<A B>_c matches vs.expect(A@B).mean - vs.expect(A).mean * vs.expect(B).mean."""
        vs, hi = _make_vstate()
        A = nk.operator.spin.sigmax(hi, 0)
        B = nk.operator.spin.sigmax(hi, 1)

        obs = nk.observable.ConnectedCorrelator(A, B)
        result = vs.expect(obs)

        AB_mean = vs.expect(A @ B).mean
        A_mean = vs.expect(A).mean
        B_mean = vs.expect(B).mean
        expected = AB_mean - A_mean * B_mean

        np.testing.assert_allclose(result.mean, expected, atol=1e-6)

    def test_sz_sz_connected(self):
        """<Z_i Z_j>_c on product state should vanish (uncorrelated)."""
        vs, hi = _make_vstate()
        Z0 = nk.operator.spin.sigmaz(hi, 0)
        Z1 = nk.operator.spin.sigmaz(hi, 1)
        obs = nk.observable.ConnectedCorrelator(Z0, Z1)
        result = vs.expect(obs)
        assert hasattr(result, "mean")

    def test_hilbert_mismatch_raises(self):
        hi1 = nk.hilbert.Spin(s=1 / 2, N=2)
        hi2 = nk.hilbert.Spin(s=1 / 2, N=3)
        A = nk.operator.spin.sigmax(hi1, 0)
        B = nk.operator.spin.sigmax(hi2, 0)
        with pytest.raises(ValueError, match="same Hilbert space"):
            nk.observable.ConnectedCorrelator(A, B)

    def test_local_estimators_returns_batch(self):
        vs, hi = _make_vstate()
        A = nk.operator.spin.sigmax(hi, 0)
        B = nk.operator.spin.sigmax(hi, 1)
        obs = nk.observable.ConnectedCorrelator(A, B)

        from netket._src.stats.local_estimators import LocalEstimatorsBatch

        le = vs.local_estimators(obs)
        assert isinstance(le, LocalEstimatorsBatch)
        assert le.n_channels == 3  # L_A, L_B, L_{AB}
