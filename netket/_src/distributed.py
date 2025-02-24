from functools import lru_cache

import numpy as np

import jax
import jax.numpy as jnp

from netket import config as nkconfig
from netket.utils import mpi
from jax.lax import with_sharding_constraint
from jax.sharding import PositionalSharding


@lru_cache
def mode() -> str:
    """
    Returns the distributed mode used by NetKet.

    This can be one of the following: ``None``, ``"sharding"``, or ``"mpi"``.
    """
    if nkconfig.netket_experimental_sharding:
        return "sharding"
    elif process_count() > 1:
        return "mpi"
    else:
        return None


@lru_cache
def process_count() -> int:
    """
    Returns the total number of JAX processes running NetKet.

    If you are running with experimental sharding, this is
    equivalent to ``jax.process_count()``. If you are running
    with mpi, this is ``nk.utils.mpi.n_nodes``.
    """

    if nkconfig.netket_experimental_sharding:
        return jax.process_count()
    else:
        return mpi.n_nodes


@lru_cache
def device_count() -> int:
    """
    Returns total number of devices.
    """
    if mode() == "sharding":
        return jax.device_count()
    else:
        return process_count()


@lru_cache
def process_index() -> int:
    """
    Returns the index of this process running NetKet.

    If you are running with experimental sharding, this is
    equivalent to ``jax.process_index()``. If you are running
    with mpi, this is ``nk.utils.mpi.rank``.

    This is an integer between 0 and
    :func:`netket_pro.distributed.process_count()`.
    """

    if nkconfig.netket_experimental_sharding:
        return jax.process_index()
    else:
        return mpi.rank


def is_master_process() -> bool:
    """
    Returns whether the current process is the master process.
    """
    return process_index() == 0


def shard_replicated(array, *, axis=0):
    """
    Shards a replicated array across MPI ranks/jax processes.

    The input must be a replicated array, obtained either from
    :func:`netket_pro.distributed.broadcast`, :func:`netket_pro.distributed.allgather` or
    from executing the same function on all nodes.

    When running under MPI, the output is simply a slice of the input array along the
    specified axis (Default 0) corresponding to the rank of the process.

    When running under sharding, we set the sharding constraint accordingly.

    Args:
        array: The array to shard. Must be replicated!
        axis: The axis along which to shard (Default 0).
    """

    def _shard(array):
        lenght = array.shape[axis]
        if not lenght % process_count() == 0:
            raise ValueError(
                "Sharded axis size must be a multiple of the number of processes"
            )

        if mode() == "sharding":
            # Do not use process_count() because we could have more than
            # 1 GPU per process

            sharding_shape = [1 for _ in range(array.ndim)]
            sharding_shape[axis] = len(jax.devices())
            sharding = jax.sharding.PositionalSharding(jax.devices()).reshape(
                sharding_shape
            )
            array = jax.lax.with_sharding_constraint(array, sharding)
        elif mode() == "mpi":
            lenght_per_proc = lenght // mpi.n_nodes
            start, end = mpi.rank * lenght_per_proc, (mpi.rank + 1) * lenght_per_proc
            array = array[start:end]
        else:
            pass
        return array

    return jax.tree_util.tree_map(_shard, array)


def allgather(array, *, axis: int = 0, token=None):
    """
    Gathers (unshard) a distributed (sharded) array to all processes.

    The resulting array will have the same shape as the input array except
    the first axis, which will be :ref:`netket_pro.distributed.process_count`
    times longer.

    .. note::

        An input array of shape :math:`(M, N, ...)` will lead to a gathered
        array of shape :math:`(P \times M, N, ...)`, where :math:`P` is the
        number of processes.

    .. note::

        The resulting array will be unsharded, or fully addressable locally
        and on every process.

    Args:
        array: The array to gather.
        token: A token to be used for MPI communication.

    Returns:
        A tuple of the gathered array and the token.

    """
    if axis != 0:
        raise NotImplementedError("Only axis=0 is supported for now. Open a PR.")

    if mode() == "mpi":
        _a = array
        array, token = mpi.mpi_allgather_jax(array, token=token)
        array = jax.lax.collapse(array, 0, 2)
    elif mode() == "sharding":
        sharding = PositionalSharding(jax.devices()).replicate()
        sharding = sharding.reshape(tuple(1 for _ in range(array.ndim)))
        array = jax.lax.with_sharding_constraint(array, sharding)
    else:
        pass
    return array, token


def pad_axis_for_sharding(
    array: jax.Array, *, axis: int = 0, padding_value: float | jax.Array = 0
) -> jax.Array:
    """
    Pads an array along an axis to make it divisible by the number of processes.

    Args:
        array: The array to pad.
        axis: The axis along which to pad.
        padding_value: The value to use for padding.

    Returns:
        The padded array.
    """
    axis_size = array.shape[axis]
    n_devices = device_count()

    if axis_size % n_devices != 0:
        padded_axis_size = int(n_devices * np.ceil(axis_size / n_devices))
        padding_shape = [(0, 0) for _ in range(array.ndim)]
        padding_shape[axis] = (0, padded_axis_size - axis_size)

        array = jnp.pad(
            array,
            padding_shape,
            constant_values=padding_value,
        )
    return array


def reshard(
    array: jax.Array,
    *,
    sharded_axis: int = 0,
    out_sharded_axis: int = 1,
    token=None,
    pad: bool = False,
    pad_value: jax.Array = 0.0,
) -> jax.Array:
    """
    Reshards an array to distribute another axis among the processes. Equivalent to
    :ref:`~mpi4jax.mpi_alltoall` in MPI jargon.

    The input array is assumed to be sharded along axis `sharded_axis`, and the resulting
    array will be sharded along axis `out_sharded_axis`. The sharded axis will be collected
    while the output sharded axis will be distributed.

    .. note::

        If the input array has shape :math:`(x, y, z)` and the input sharded axis is `y`,
        and the output sharded axis is `x`, the resulting array will have shape :math:`(x, y*P, z/P)`.

    Args:
        array: The array to reshard / alltoall.
        sharded_axis: The axis to be collected.
        out_sharded_axis: The axis to be distributed.
        token: A token to be used for MPI communication.
        pad: Whether to pad the axis to be sharded to be a multiple of the number of processes. If this is
            set to `False`, the size of the sharded axis must be a multiple of the number of processes.
            (Default: `False`)
        pad_value: The value to use for padding. (Default: `0.0`)

    """
    assert sharded_axis != out_sharded_axis
    assert 0 <= sharded_axis < array.ndim
    assert 0 <= out_sharded_axis < array.ndim

    # Pad the number of parameters to be a multiple of the number of MPI nodes
    # -> (#n_nodes, np_padded)
    if array.shape[out_sharded_axis] % device_count() != 0:
        if pad:
            array = pad_axis_for_sharding(
                array, axis=out_sharded_axis, padding_value=pad_value
            )
        else:
            raise ValueError(
                "Sharded axis size must be a multiple of the number of processes"
            )

    if mode() == "mpi":
        # Create a new shape with the axis to be sharded split into two
        # (..., M, ...) -> (..., n, M/n, ...)
        new_shape = list(array.shape)
        new_shape.insert(out_sharded_axis, process_count())
        new_shape[out_sharded_axis + 1] = -1
        array = jnp.reshape(array, new_shape)

        # Move this axis to the position 0
        # (..., n, M/n, ...) -> (n, ..., M/n, ...)
        array = jnp.moveaxis(array, out_sharded_axis, 0)
        array, token = mpi.mpi_alltoall_jax(array, token=token)

        # After the alltoall, the sharded axis is not split between the
        # position 0 and the actual sharded axis, so we need to collapse them.
        # First we move the sharded axis back to its original position
        if sharded_axis != 0:
            array = jnp.moveaxis(array, 0, sharded_axis)

        # Then we collapse them
        array = jax.lax.collapse(array, sharded_axis, sharded_axis + 2)
    elif mode() == "sharding":
        del sharded_axis  # unused

        sharding = PositionalSharding(jax.devices())
        sharding_shape = list(1 for _ in range(array.ndim))
        sharding_shape[out_sharded_axis] = -1
        sharding = sharding.reshape(sharding_shape)
        array = with_sharding_constraint(array, sharding)
    return array, token
