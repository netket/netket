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

import jax
from jax import numpy as jnp

from netket.hilbert import DoubledHilbert
from netket.utils.dispatch import dispatch

from ..homogeneous import HomogeneousHilbert
from .base import flip_state_scalar, random_state


@dispatch
def random_state(hilb: DoubledHilbert, key, batches: int, *, dtype):  # noqa: F811
    key1, key2 = jax.random.split(key)

    v1 = random_state(hilb.physical, key1, batches, dtype)
    v2 = random_state(hilb.physical, key2, batches, dtype)

    return jnp.concatenate([v1, v2], axis=-1)


@dispatch
def flip_state_scalar(hilb: DoubledHilbert, key, state, index):  # noqa: F811
    return _flip_state_scalar_fallback(hilb, key, state, index)


# If homogeneous with no constraint, use faster implementation
# and do not consider it as a doubled hilbert.
@dispatch
def flip_state_scalar(  # noqa: F811
    hilb: DoubledHilbert[HomogeneousHilbert], key, state, index
):
    if not hilb.physical.constrained:
        return flip_state_scalar(hilb.physical, key, state, index)
    else:
        return _flip_state_scalar_fallback(hilb, key, state, index)


# default implementation
def _flip_state_scalar_fallback(hilb, key, state, index):
    def flip_lower_state_scalar(args):
        key, state, index = args
        return flip_state_scalar(hilb.physical, key, state, index)

    def flip_upper_state_scalar(args):
        key, state, index = args
        return flip_state_scalar(hilb.physical, key, state, index - hilb.size)

    return jax.lax.cond(
        index < hilb.physical.size,
        flip_lower_state_scalar,
        flip_upper_state_scalar,
        (key, state, index),
    )
