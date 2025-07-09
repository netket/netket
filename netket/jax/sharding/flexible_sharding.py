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

"""
Internal utility functions to support jax sharding natively within netket.
All functions in here are not part of the public API, internal, and may change without warning.
"""

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp


safe_zip = partial(zip, strict=True)
_identity = lambda x: x


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
    n_devices = jax.device_count()

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
