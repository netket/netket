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

from typing import Optional, List

import jax
import numpy as np
from jax import numpy as jnp

# from numba import jit

from netket.hilbert import DoubledHilbert

from .base import (
    random_state_batch,
    register_random_state_impl,
    flip_state_scalar,
    register_flip_state_impl,
)


def random_state_batch_doubled_impl(hilb: DoubledHilbert, key, batches, dtype):
    shape = (batches, hilb.size)

    key1, key2 = jax.random.split(key)

    v1 = random_state_batch(hilb.physical, key1, batches, dtype)
    v2 = random_state_batch(hilb.physical, key2, batches, dtype)

    return jnp.concatenate([v1, v2], axis=1)


## flips
def flip_state_scalar_doubled(hilb: DoubledHilbert, key, state, index):
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


register_random_state_impl(DoubledHilbert, batch=random_state_batch_doubled_impl)
register_flip_state_impl(DoubledHilbert, scalar=flip_state_scalar_doubled)
