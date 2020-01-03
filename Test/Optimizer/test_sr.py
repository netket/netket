import netket as nk
from netket.optimizer import SR

import numpy as np

import pytest


def test_sr_no_segfault():
    """
    Tests the resolution of bug #317.
    """
    sr = SR()
    assert sr.last_covariance_matrix is None


def test_svd_threshold():
    """
    Test SVD threshold option of BDCSVD
    """
    with pytest.raises(
        ValueError, match="svd_threshold option only available for BDCSVD solver"
    ):
        SR(svd_threshold=1e-3)
    with pytest.raises(
        ValueError, match="svd_threshold option only available for BDCSVD solver"
    ):
        SR(use_iterative=True, svd_threshold=1e-3)
    with pytest.raises(
        ValueError, match="svd_threshold option only available for BDCSVD solver"
    ):
        SR(lsq_solver="LDLT", svd_threshold=1e-3)

    a = np.diag([1e0 + 0j, 1e-3, 1e-6])
    b = np.array([1.0 + 0j, 1.0, 1.0])

    def SR_with_threshold(t):
        return SR(lsq_solver="BDCSVD", svd_threshold=t, diag_shift=0)

    def solve(sr, a, b):
        a1 = np.sqrt(a) * np.sqrt(a.shape[0])
        out = np.copy(b)
        sr.compute_update(a1, b, out)
        return out

    sr = SR_with_threshold(1e-1)
    assert np.allclose(solve(sr, a, b), [1.0, 0.0, 0.0])

    sr = SR_with_threshold(1e-3)
    assert np.allclose(solve(sr, a, b), [1.0, 1e3, 0.0])

    sr = SR_with_threshold(1e-6)
    assert np.allclose(solve(sr, a, b), [1.0, 1e3, 1e6])
