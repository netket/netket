import numpy as np
import pytest

import jax
import jax.numpy as jnp

from netket.utils import mpi


def approx(data):
    return pytest.approx(data, abs=1.0e-6, rel=1.0e-5)


def test_gather_jax():
    rank = mpi.rank
    size = mpi.n_nodes

    arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: mpi.mpi_gather_jax(x, root=0)[0])(arr)
    if rank == 0:
        for p in range(size):
            np.testing.assert_allclose(res[p], jnp.ones((3, 2)) * p)
    else:
        np.testing.assert_allclose(res, arr)


def test_allgather_jax():
    rank = mpi.rank
    size = mpi.n_nodes

    arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: mpi.mpi_allgather_jax(x)[0])(arr)
    for p in range(size):
        np.testing.assert_allclose(res[p], jnp.ones((3, 2)) * p)


def test_scatter_jax():
    rank = mpi.rank
    size = mpi.n_nodes

    if rank == 0:
        arr = jnp.stack([jnp.ones((3, 2)) * r for r in range(size)], axis=0)
    else:
        arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: mpi.mpi_scatter_jax(x, root=0)[0])(arr)

    np.testing.assert_allclose(res, jnp.ones((3, 2)) * rank)


def test_alltoall_jax():
    rank = mpi.rank
    size = mpi.n_nodes

    arr = jnp.ones((size, 3, 2)) * rank

    res = jax.jit(lambda x: mpi.mpi_alltoall_jax(x)[0])(arr)
    for p in range(size):
        np.testing.assert_allclose(res[p], jnp.ones((3, 2)) * p)


def test_reduce_jax():
    rank = mpi.rank
    size = mpi.n_nodes

    arr = jnp.ones((3, 2)) * rank

    res = jax.jit(lambda x: mpi.mpi_reduce_sum_jax(x, root=0)[0])(arr)
    if rank == 0:
        np.testing.assert_allclose(res, jnp.ones((3, 2)) * sum(range(size)))
    else:
        np.testing.assert_allclose(res, arr)
