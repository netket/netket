# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .mpi import n_nodes, MPI, MPI_py_comm, MPI_jax_comm


def mpi_sum(x, *, comm=MPI_py_comm):
    """
    Computes the elementwise sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    ar = np.asarray(x)

    if n_nodes > 1:
        comm.Allreduce(MPI.IN_PLACE, ar.reshape(-1), op=MPI.SUM)

    return ar


def mpi_sum_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm, token=token)


def mpi_prod(x, *, comm=MPI_py_comm):
    """
    Computes the elementwise sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    ar = np.asarray(x)

    if n_nodes > 1:
        comm.Allreduce(MPI.IN_PLACE, ar.reshape(-1), op=MPI.PROD)

    return ar


def mpi_prod_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.PROD, comm=comm, token=token)


def mpi_mean(x, *, comm=MPI_py_comm):
    """
    Computes the elementwise mean of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    return mpi_sum(x, comm=comm) / n_nodes


def mpi_mean_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise mean of an array or a scalar across all MPI processes
    of a jax array.

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    res, token = mpi_sum_jax(x, token=token, comm=comm)
    return res / n_nodes, token


def mpi_any(x, *, comm=MPI_py_comm):
    """
    Computes the elementwise logical OR of an array or a scalar across all MPI
    processes, effectively equivalent to an elementwise any

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    ar = np.asarray(x)

    if n_nodes > 1:
        comm.Allreduce(MPI.IN_PLACE, ar.reshape(-1), op=MPI.LOR)

    return ar


def mpi_any_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise logical OR of an array or a scalar across all MPI
    processes, effectively equivalent to an elementwise any

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.LOR, comm=comm, token=token)


def mpi_all(x, *, comm=MPI_py_comm):
    """
    Computes the elementwise logical AND of an array or a scalar across all MPI
    processes.

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    ar = np.asarray(x)

    if n_nodes > 1:
        comm.Allreduce(MPI.IN_PLACE, ar.reshape(-1), op=MPI.LAND)

    return ar


def mpi_all_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise logical AND of an array or a scalar across all MPI
    processes.

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.LAND, comm=comm, token=token)


def mpi_max(x, *, comm=MPI_py_comm):
    """
    Computes the elementwise logical OR of an array or a scalar across all MPI
    processes, effectively equivalent to an elementwise any

    Args:
        a: The input array, which will usually be overwritten in place.
    Returns:
        out: The reduced array.
    """
    ar = np.asarray(x)

    if n_nodes > 1:
        comm.Allreduce(MPI.IN_PLACE, ar.reshape(-1), op=MPI.MAX)

    return ar


def mpi_max_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise logical OR of an array or a scalar across all MPI
    processes, effectively equivalent to an elementwise any

    Args:
        a: The input array.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.MAX, comm=comm, token=token)


def mpi_bcast(x, *, root, comm=MPI_py_comm):
    if n_nodes == 1:
        return x
    else:
        return comm.bcast(x, root=root)


def mpi_bcast_jax(x, *, token=None, root, comm=MPI_jax_comm):
    if n_nodes == 1:
        assert root == 0
        return x, token
    else:
        import mpi4jax

        return mpi4jax.bcast(x, token=token, root=root, comm=comm)


def mpi_allgather(x, *, comm=MPI_py_comm):
    if n_nodes == 1:
        return x
    else:
        return comm.allgather(x)


def mpi_allgather_jax(x, *, token=None, comm=MPI_jax_comm):
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allgather(x, token=token, comm=comm)
