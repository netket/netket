import netket as nk
import numpy as np
from mpi4py import MPI

import pytest
from pytest import approx

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


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

    Rhat = np.sqrt((N - 1) / N + var_between / (N * var_mean))

    return mean_full, var_full, Rhat


def test_mc_stats():
    # Test data of shape [MPI_size, n_chains, n_samples], same on all ranks
    data = np.random.rand(size, 10, 1000)
    data = comm.bcast(data)

    ref_mean, ref_var, ref_R = reference_stats(data)

    mydata = data[rank]

    stats = nk.stats.statistics(mydata)

    assert stats.mean == approx(ref_mean)
    assert stats.variance == approx(ref_var)
    assert stats.R == approx(ref_R)


def test_mean():
    data = np.random.rand(size, 10, 11, 12)
    data = comm.bcast(data)
    mydata = data[rank]

    for axis in None, 0, 1, 2:
        ref_mean = np.mean(data.mean(0), axis=axis)
        nk_mean = nk.stats.mean(mydata, axis=axis)

        assert nk_mean.shape == ref_mean.shape, "axis={}".format(axis)
        assert nk_mean == approx(ref_mean), "axis={}".format(axis)


def test_var():
    data = np.random.rand(size, 10, 11, 12)
    data = comm.bcast(data)
    mydata = data[rank]

    for axis in None, 0, 1, 2:
        if axis is not None:
            # Merge first "MPI axis" with target axis
            refdata = np.moveaxis(data, axis + 1, 0)
            refdata = refdata.reshape(-1, *refdata.shape[2:])
            # Compute variance over merged axis
            ref_var = np.var(refdata, axis=0, ddof=0)
        else:
            ref_var = np.var(data, ddof=0)

        nk_var = nk.stats.var(mydata, axis=axis)

        assert nk_var.shape == ref_var.shape, "axis={}".format(axis)
        assert nk_var == approx(ref_var), "axis={}".format(axis)


def test_sum():
    data = np.arange(size * 10 * 11).reshape(size, 10, 11)
    data = comm.bcast(data)
    mydata = data[rank]

    ref_sum = np.sum(data, axis=0)
    ret = nk.stats.mpi_sum_inplace(mydata)
    # mydata should be changed in place
    assert mydata == approx(ref_sum)
    assert np.all(ret == mydata)
