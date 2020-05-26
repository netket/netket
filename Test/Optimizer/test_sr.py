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
        ValueError,
        match="The svd_threshold option is available only for non-sparse solvers.",
    ):
        SR(use_iterative=True, svd_threshold=1e-3)
