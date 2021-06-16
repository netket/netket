import numpy as np
import pytest

import jax
import jax.numpy as jnp

from functools import partial

from netket.stats import statistics
from scipy.optimize import curve_fit
from numba import jit

import netket as nk

from .. import common

WEIGHT_SEED = 3


@pytest.fixture()
def arr(request, _mpi_size):
    return jax.random.normal(jax.random.PRNGKey(WEIGHT_SEED), (100 * _mpi_size, 10))


@pytest.fixture()
def arr_loc(request, arr, _mpi_rank):
    return arr[100 * _mpi_rank : 100 * (_mpi_rank + 1), :]


@pytest.mark.parametrize("axis", [None, 0])
def test_mean(axis, arr, arr_loc):
    arr_mean = jnp.mean(arr, axis=axis)

    nk_mean = nk.stats.mean(arr_loc, axis=axis)
    np.testing.assert_allclose(nk_mean, arr_mean)

    nk_mean = nk.stats.mean(arr_loc, keepdims=True, axis=axis)
    assert nk_mean.ndim == arr.ndim
    np.testing.assert_allclose(nk_mean, arr_mean.reshape(1, -1))


@pytest.mark.parametrize("axis", [None, 0])
def test_subtract_mean(axis, arr, arr_loc, _mpi_rank):
    arr_sub = arr - jnp.mean(arr, axis=axis)

    nk_sub = nk.stats.subtract_mean(arr_loc, axis=axis)
    np.testing.assert_allclose(
        nk_sub, arr_sub[100 * _mpi_rank : 100 * (_mpi_rank + 1), :]
    )


@pytest.mark.parametrize("axis", [None, 0])
def test_sum(axis, arr, arr_loc):
    arr_sum = jnp.sum(arr, axis=axis)

    nk_sum = nk.stats.sum(arr_loc, axis=axis)
    np.testing.assert_allclose(nk_sum, arr_sum)

    nk_sum = nk.stats.sum(arr_loc, keepdims=True, axis=axis)
    assert nk_sum.ndim == arr.ndim
    np.testing.assert_allclose(nk_sum, arr_sum.reshape(1, -1))


@pytest.mark.parametrize("axis", [None, 0])
def test_var(axis, arr, arr_loc):
    arr_var = jnp.var(arr, axis=axis)

    nk_var = nk.stats.var(arr_loc, axis=axis)
    np.testing.assert_allclose(nk_var, arr_var)


@pytest.mark.parametrize("axis", [None, 0])
def test_total_size(axis, arr, arr_loc, _mpi_size):
    if axis is None:
        sz = arr.size
    else:
        sz = arr.shape[axis]
    sz = sz

    nk_sz = nk.stats.total_size(arr_loc, axis=axis)
    assert nk_sz == sz
