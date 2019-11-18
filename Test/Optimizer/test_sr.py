import netket as nk

def test_sr_no_segfault():
    """
    Tests the resolution of bug #317.
    """
    sr = nk.optimizer.SR()
    assert sr.last_covariance_matrix is None

