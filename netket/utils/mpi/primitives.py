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

from typing import Any
from functools import wraps

import numpy as np
import jax
import jax.numpy as jnp

from netket.utils.types import Array

from .mpi import n_nodes, MPI, MPI_py_comm, MPI_jax_comm

Token = Any


def promote_to_pytree(f):
    """
    Decorator for an mpi4jax function to make it work with pytrees.

    Args:
        f: A function that takes an array and an optional token argument,
            and returns an array and a token.

    Returns:
        A function that takes a pytree and an optional token argument,
    """

    @wraps(f)
    def f_pytree(pytree, *args, token=None, **kwargs):
        pytree_flat, pytree_struct = jax.tree_util.tree_flatten(pytree)
        output_flat = []
        for arr in pytree_flat:
            output, token = f(arr, *args, token=token, **kwargs)
            output_flat.append(output)

        output_pytree = jax.tree_util.tree_unflatten(pytree_struct, output_flat)
        return output_pytree, token

    return f_pytree


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


@promote_to_pytree
def mpi_sum_jax(
    x: Array, *, token: Token = None, comm=MPI_jax_comm
) -> tuple[jax.Array, Token]:
    """
    Computes the elementwise sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    .. note::
        This function supports JAX pytrees, in which case the reduction is performed
        on every leaf of the pytree.

    Args:
        a: The input array or pytree.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return jnp.asarray(x), token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm, token=token)  # type: ignore


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


@promote_to_pytree
def mpi_prod_jax(
    x: Array, *, token: Token = None, comm=MPI_jax_comm
) -> tuple[jax.Array, Token]:
    """
    Computes the elementwise sum of an array or a scalar across all MPI processes.
    Attempts to perform this sum inplace if possible, but for some types a copy
    might be returned.

    .. note::
        This function supports JAX pytrees, in which case the reduction is performed
        on every leaf of the pytree.

    Args:
        a: The input array or pytree.
        token: An optional token to impose ordering of MPI operations

    Returns:
        out: The reduced array.
        token: an output token
    """
    if n_nodes == 1:
        return jnp.asarray(x), token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.PROD, comm=comm, token=token)  # type: ignore


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


@promote_to_pytree
def mpi_mean_jax(
    x: Array, *, token: Token = None, comm=MPI_jax_comm
) -> tuple[jax.Array, Token]:
    """
    Computes the elementwise mean of an array or a scalar across all MPI processes
    of a jax array.

    .. note::
        This function supports JAX pytrees, in which case the reduction is performed
        on every leaf of the pytree.

    Args:
        a: The input array or pytree.
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


@promote_to_pytree
def mpi_any_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise logical OR of an array or a scalar across all MPI
    processes, effectively equivalent to an elementwise any

    .. note::
        This function supports JAX pytrees, in which case the reduction is performed
        on every leaf of the pytree.


    Args:
        a: The input array or pytree.
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


@promote_to_pytree
def mpi_all_jax(x, *, token=None, comm=MPI_jax_comm):
    """
    Computes the elementwise logical AND of an array or a scalar across all MPI
    processes.

    .. note::
        This function supports JAX pytrees, in which case the reduction is performed
        on every leaf of the pytree.

    Args:
        a: The input array or pytree.
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


@promote_to_pytree
def mpi_max_jax(
    x: Array, *, token: Token = None, comm=MPI_jax_comm
) -> tuple[jax.Array, Token]:
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
        return jnp.asarray(x), token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.MAX, comm=comm, token=token)  # type: ignore


def mpi_bcast(x, *, root, comm=MPI_py_comm):
    if n_nodes == 1:
        return x
    else:
        return comm.bcast(x, root=root)


@promote_to_pytree
def mpi_bcast_jax(
    x: Array, *, token: Token = None, root, comm=MPI_jax_comm
) -> tuple[jax.Array, Token]:
    if n_nodes == 1:
        assert root == 0
        return jnp.asarray(x), token
    else:
        import mpi4jax

        return mpi4jax.bcast(x, token=token, root=root, comm=comm)  # type: ignore


def mpi_allgather(x, *, comm=MPI_py_comm):
    if n_nodes == 1:
        if isinstance(x, np.ndarray | jax.Array):
            return x.reshape(1, *x.shape)
        else:
            return (x,)
    else:
        if isinstance(x, np.ndarray | jax.Array):
            out = np.empty((n_nodes,) + x.shape, dtype=x.dtype)
            comm.Allgather(np.asarray(x), out)
            return out
        else:
            return comm.allgather(x)


def mpi_gather(x, *, root: int = 0, comm=MPI_py_comm):
    if n_nodes == 1:
        if isinstance(x, np.ndarray | jax.Array):
            return x.reshape(1, *x.shape)
        else:
            return (x,)
    else:
        if isinstance(x, np.ndarray | jax.Array):
            out = np.empty((n_nodes,) + x.shape, dtype=x.dtype)
            comm.Gather(np.asarray(x), out, root=root)
            return out
        else:
            return comm.gather(x, root=root)


@promote_to_pytree
def mpi_gather_jax(
    x: Array, *, token: Token = None, root: int = 0, comm=MPI_jax_comm
) -> tuple[jax.Array, Token]:
    if n_nodes == 1:
        return jnp.expand_dims(x, 0), token
    else:
        import mpi4jax

        return mpi4jax.gather(x, token=token, root=root, comm=comm)  # type: ignore


@promote_to_pytree
def mpi_allgather_jax(
    x: Array, *, token: Token = None, comm=MPI_jax_comm
) -> tuple[jax.Array, Token]:
    if n_nodes == 1:
        return jnp.expand_dims(x, 0), token
    else:
        import mpi4jax

        return mpi4jax.allgather(x, token=token, comm=comm)  # type: ignore


@promote_to_pytree
def mpi_scatter_jax(
    x: Array, *, token: Token = None, root: int = 0, comm=MPI_jax_comm
) -> tuple[jax.Array, Token]:
    if n_nodes == 1:
        if x.shape[0] != 1:
            raise ValueError("Scatter input must have shape (nproc, ...)")
        return x[0], token
    else:
        import mpi4jax

        return mpi4jax.scatter(x, root=root, token=token, comm=comm)  # type: ignore


@promote_to_pytree
def mpi_alltoall_jax(x, *, token=None, comm=MPI_jax_comm):
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.alltoall(x, token=token, comm=comm)


@promote_to_pytree
def mpi_reduce_sum_jax(x, *, token=None, root: int = 0, comm=MPI_jax_comm):
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.reduce(x, op=MPI.SUM, root=root, token=token, comm=comm)


@promote_to_pytree
def mpi_allreduce_sum_jax(x, *, token=None, root: int = 0, comm=MPI_jax_comm):
    if n_nodes == 1:
        return x, token
    else:
        import mpi4jax

        return mpi4jax.allreduce(x, op=MPI.SUM, token=token, comm=comm)
