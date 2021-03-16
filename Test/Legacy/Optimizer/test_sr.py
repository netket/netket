import netket.legacy as nk
from netket.legacy.optimizer import SR

import numpy as np

import pytest


def test_sr_no_segfault():
    """
    Tests the resolution of bug #317.
    """
    # Hilbert space of spins on the graph
    hi = nk.hilbert.Spin(s=0.5) ** 20
    machine = nk.machine.RbmSpin(alpha=1, hilbert=hi)
    sr = SR(machine)
    assert sr.last_covariance_matrix is None


def test_svd_threshold():
    """
    Test SVD threshold option of BDCSVD
    """
    with pytest.raises(
        ValueError,
        match="The svd_threshold option is available only for non-sparse solvers.",
    ):
        # Hilbert space of spins on the graph
        hi = nk.hilbert.Spin(s=0.5) ** 20
        machine = nk.machine.RbmSpin(alpha=1, hilbert=hi)
        SR(machine, use_iterative=True, svd_threshold=1e-3)
