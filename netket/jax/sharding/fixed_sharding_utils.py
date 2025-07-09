# Copyright 2025 The NetKet Authors - All rights reserved.
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

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

from netket.utils import config
from netket.utils.deprecation import warn_deprecation


safe_zip = partial(zip, strict=True)
_identity = lambda x: x


@partial(jax.jit, static_argnums=(0, 1))
def _prepare_mask(n, n_pad):
    return jnp.ones(n + n_pad, dtype=bool).at[-n_pad:].set(0)


def distribute_to_devices_along_axis(
    inp_data, axis=0, pad: bool = False, pad_value=None, devices=None
):
    """
    Distribute a local array equally along an axis to multiple jax devices devices

     .. note:
        Does nothing if netket.config.netket_experimental_sharding=False.

     .. note:
        Each jax process needs to have the whole array (parts not belonging to it can be filled with garbage).

    Args:
        inp_data: the full array (on every process)
        axis: (optional) axis along which to distribute
        pad: If True: pad the input data along axis to the next multiple of the number of devices
              If False (default): no padding; the size of the axis in inp_data needs to be divisible by the number of devices.
        pad_value: value to pad with (optional, only used if pad=True)
        devices: (optional) list of jax devices. Defaults to all available devices

    Returns:
        out_data: a distributed jax.Array
        mask: a mask indicating wether a given element is part of the original data (True) of of the padding (False)
              only returned if pad=True
    """
    if devices is None:
        devices = jax.devices()

    if config.netket_experimental_sharding:
        if pad:
            n = inp_data.shape[0]
            # pad to the next multiple of device_count
            device_count = jax.device_count()
            n_pad = math.ceil(inp_data.shape[0] / device_count) * device_count - n
            inp_data = jnp.pad(inp_data, ((0, n_pad), (0, 0)))
            if pad_value is not None and n_pad > 0:
                inp_data = inp_data.at[-n_pad:].set(pad_value)

        shape = [
            None,
        ] * inp_data.ndim
        shape[axis] = "S"
        mesh = jax.sharding.get_abstract_mesh()
        sharding = NamedSharding(mesh, P(*shape))
        out_data = jax.jit(_identity, out_shardings=sharding)(inp_data)

        if pad:
            if n_pad > 0:  # type: ignore
                mask = _prepare_mask(n, n_pad)  # type: ignore
            else:
                mask = None
            return out_data, mask
        else:
            return out_data
    else:
        return inp_data


# TODO consider merging this with distribute_to_devices_along_axis
@partial(jax.jit, static_argnames="axis")
def shard_along_axis(x, axis: int):
    """
    When running with experimental sharding mode, calls
    :func:`jax.lax.with_sharding_constraint` with a
    :class:`jax.sharding.NamedSharding` sharded along 'S' the given axis.

    Args:
        x: An array
        axis: the axis to be sharded
    """
    if config.netket_experimental_sharding and jax.device_count() > 1:
        # Shard shape is (1, 1, 1, -1, 1, 1) where -1 is the axis
        shard_shape = [None for _ in range(x.ndim)]
        shard_shape[axis] = "S"

        mesh = jax.sharding.get_abstract_mesh()
        x = jax.lax.with_sharding_constraint(
            x,
            NamedSharding(mesh, P(*shard_shape)),
        )
    return x


@jax.jit
def with_samples_sharding_constraint(x, shape=None):
    """
    ensure the input x is sharded along axis 0 on all devices
    works both outside and inside of jit
    """
    warn_deprecation(
        "with_samples_sharding_constraint is deprecated in favour of nk.jax.sharding.shard_along_axis(x, axis=0)"
    )

    return shard_along_axis(x, 0)


def extract_replicated(t):
    """
    Extract the value of a fully replicated global device array.

    Args:
        t: a jax Array (or a pytree of jax Arrays)

    Returns:
        A locally adressable representation of t
    """

    def _extract_replicated(x):
        if isinstance(x, jax.Array) and not x.is_fully_addressable:
            if not x.is_fully_replicated:
                raise RuntimeError(
                    "Expected a fully replicated array, but found one that is not. You should gather the array first. If you are not developing custom logic, please open a bug report with a reproducer."
                )
            return x.addressable_data(0)
        else:
            return x

    return jax.tree_util.tree_map(_extract_replicated, t)


def gather(x):
    """
    Make a sharded array fully replicated by gathering all parts on every device.

    Args:
        x: potentially unreplicated jax.Array with NamedSharding(...'S')

    Returns:
        fully replicated array
    """

    if not isinstance(x, jax.Array):
        # TODO in the future we could chagne it to just return x unchanged
        # but for now we error if x is not a jax array to ensure gather is used correctly
        raise RuntimeError("gather can only be applied to a jax.Array")
    elif x.is_fully_replicated:  # includes SingleDeviceSharding
        return x
    elif isinstance(x.sharding, NamedSharding):
        # x.sharding.device_set has arbitrary order
        # Hardcode all devices until I figure out a way to deduce the order from x
        out_shardings = NamedSharding(jax.sharding.get_abstract_mesh(), P())
    else:
        raise NotImplementedError(
            "Gather is not compatible with {x.sharding}. Please open a feature request."
        )
    return jax.jit(_identity, out_shardings=out_shardings)(x)
