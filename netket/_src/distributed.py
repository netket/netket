from functools import lru_cache

import numpy as np

import jax
import jax.numpy as jnp

from netket import config as nkconfig


@lru_cache
def mode() -> str:
    """
    Returns the distributed mode used by NetKet.

    This can be one of the following: ``None``, ``"sharding"``, or ``"mpi"``.
    """
    if nkconfig.netket_experimental_sharding:
        return "sharding"
    else:
        return None


@lru_cache
def process_count() -> int:
    """
    Returns the total number of JAX processes running NetKet.

    If you are running with experimental sharding, this is
    equivalent to ``jax.process_count()``.
    """

    if nkconfig.netket_experimental_sharding:
        return jax.process_count()
    else:
        return 1


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
    equivalent to ``jax.process_index()``.

    This is an integer between 0 and
    :func:`netket_pro.distributed.process_count()`.
    """

    if nkconfig.netket_experimental_sharding:
        return jax.process_index()
    else:
        return 0


def is_master_process() -> bool:
    """
    Returns whether the current process is the master process.
    """
    return process_index() == 0


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


def _inspect(name: str, tree: jax.Array):

    def _inspect_(name: str, x: jax.Array):
        """
        Internal function to inspect the sharding of an array. To be used for debugging inside
        of :func:`jax.jit`-ted functions.

        Args:
            name: A string to identify the array, usually the name, but can contain anything else.
            x: The array
        """
        if mode() == "sharding":

            def _cb(y):
                if process_index() == 0:
                    print(
                        f"{name}: shape={x.shape}, sharding:",
                        y,
                        flush=True,
                    )

            jax.debug.inspect_array_sharding(x, callback=_cb)

    if isinstance(tree, jax.Array):
        _inspect_(name, tree)
    else:
        jax.tree.map_with_path(lambda path, x: _inspect_(name + str(path), x), tree)
