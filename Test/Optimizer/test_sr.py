import netket as nk
from netket.optimizer import SR

import numpy as np

import pytest


def test_sr_no_segfault():
    """
    Tests the resolution of bug #317.
    """
    # 1D Lattice
    g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

    # Hilbert space of spins on the graph
    hi = nk.hilbert.Spin(s=0.5, graph=g)
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
        # 1D Lattice
        g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

        # Hilbert space of spins on the graph
        hi = nk.hilbert.Spin(s=0.5, graph=g)
        machine = nk.machine.RbmSpin(alpha=1, hilbert=hi)
        SR(machine, use_iterative=True, svd_threshold=1e-3)
