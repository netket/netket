import itertools
import netket as nk
import numpy as np
import pytest

from .. import common


def approx(data):
    return pytest.approx(data, abs=1.0e-6, rel=1.0e-5)


def reference_stats(data):
    """
    Computes the statistics on the full data, without MPI,
    for reference.
    """
    M_par, M_loc, N = data.shape
    M_full = M_par * M_loc
    data_ = data.reshape(M_full, N)

    chain_means = np.mean(data_, axis=1)
    chain_vars = np.var(data_, axis=1, ddof=0)

    mean_full = np.mean(data)
    var_full = np.var(data, ddof=0)

    var_mean = np.mean(chain_vars)
    var_between = N * np.var(chain_means, ddof=0)

    R_hat = np.sqrt((N - 1) / N + var_between / (N * var_mean))

    return mean_full, var_full, R_hat


@common.onlyif_mpi
def test_mc_stats(_mpi_comm, _mpi_rank, _mpi_size):
    # Test data of shape [MPI__mpi_size, n_chains, n_samples], same on all _mpi_ranks
    data = np.random.rand(_mpi_size, 10, 1000)
    data = _mpi_comm.bcast(data)

    ref_mean, ref_var, ref_R = reference_stats(data)

    mydata = np.copy(data[_mpi_rank])

    stats = nk.stats.statistics(mydata)

    assert nk.stats.mean(data) == approx(ref_mean)
    assert stats.mean == approx(ref_mean)
    assert stats.variance == approx(ref_var)
    assert stats.R_hat == approx(ref_R)


@common.onlyif_mpi
def test_mean(_mpi_comm, _mpi_rank, _mpi_size):
    data = np.random.rand(_mpi_size, 10, 11, 12)
    data = _mpi_comm.bcast(data)
    mydata = np.copy(data[_mpi_rank])

    for axis in None, 0, 1, 2:
        ref_mean = np.mean(data.mean(0), axis=axis)
        nk_mean = nk.stats.mean(mydata, axis=axis)

        assert nk_mean.shape == ref_mean.shape, "axis={}".format(axis)
        assert nk_mean == approx(ref_mean), "axis={}".format(axis)

    # Test with out
    out = nk.stats.mean(mydata)
    assert out == approx(np.mean(data))

    # Test with out and axis
    out = nk.stats.mean(mydata, axis=0)
    assert out == approx(np.mean(data.mean(0), axis=0))

    # Test with complex dtype
    out = nk.stats.mean(1j * mydata, axis=0)
    assert out == approx(np.mean(1j * data.mean(0), axis=0))

    # Test with keepdims
    out = nk.stats.mean(mydata, keepdims=True)
    assert out == approx(np.mean(data.mean(0), keepdims=True))


@common.onlyif_mpi
def test_sum(_mpi_comm, _mpi_rank, _mpi_size):
    data = np.ones((_mpi_size, 10, 11, 12))
    data = _mpi_comm.bcast(data)
    mydata = np.copy(data[_mpi_rank])

    for axis in None, 0, 1, 2:
        ref_sum = np.sum(data.sum(axis=0), axis=axis)
        nk_sum = nk.stats.sum(mydata, axis=axis)
        nk_sum_kd = nk.stats.sum(mydata, axis=axis, keepdims=True)
        ref_sum_kd = np.sum(data.sum(axis=0), axis=axis, keepdims=True)

        assert nk_sum.shape == ref_sum.shape, "axis={}".format(axis)
        assert np.all(nk_sum == ref_sum), "axis={}".format(axis)
        assert nk_sum_kd.shape == ref_sum_kd.shape, "axis={}, keepdims=True".format(
            axis
        )
        assert np.all(nk_sum_kd == ref_sum_kd), "axis={}, keepdims=True".format(axis)

    # Test with out
    out = nk.stats.sum(mydata)
    np.testing.assert_almost_equal(out, np.sum(data))

    # Test with out and axis
    out = nk.stats.sum(mydata, axis=0)
    np.testing.assert_almost_equal(out, np.sum(data.sum(axis=0), axis=0))

    # Test with complex dtype
    out = nk.stats.sum(1j * mydata, axis=0)
    np.testing.assert_almost_equal(out, np.sum(1j * data.sum(axis=0), axis=0))


@common.onlyif_mpi
def test_var(_mpi_comm, _mpi_rank, _mpi_size):
    data = np.random.rand(_mpi_size, 10, 11, 12, 13)
    data = _mpi_comm.bcast(data)
    mydata = np.copy(data[_mpi_rank])

    for axis, ddof in itertools.product((None, 0, 1, 2, 3), (0, 1)):
        if axis is not None:
            # Merge first "MPI axis" with target axis
            refdata = np.moveaxis(data, axis + 1, 0)
            refdata = refdata.reshape(-1, *refdata.shape[2:])
            # Compute variance over merged axis
            ref_var = np.var(refdata, axis=0, ddof=ddof)
        else:
            ref_var = np.var(data, ddof=ddof)

        nk_var = nk.stats.var(mydata, axis=axis, ddof=ddof)

        assert nk_var.shape == ref_var.shape, "axis={},ddof={}".format(axis, ddof)
        assert nk_var == approx(ref_var), "axis={},ddof={}".format(axis, ddof)

    # Test with out
    out = nk.stats.var(mydata)
    assert out == approx(np.var(data))


@common.onlyif_mpi
def test_subtract_mean(_mpi_rank, _mpi_size, _mpi_comm):
    data = np.random.rand(_mpi_size, 10, 11, 12)
    data = _mpi_comm.bcast(data)
    mydata = np.copy(data[_mpi_rank])

    ref_mean = nk.stats.mean(mydata, axis=0)
    ref_data = mydata - ref_mean[np.newaxis, :, :]

    mydata = nk.stats.subtract_mean(mydata, axis=0)
    assert mydata == approx(ref_data)
