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

from netket.hilbert import TensorHilbert

from .base import (
    random_state_batch,
    register_random_state_impl,
    flip_state_scalar,
    register_flip_state_impl,
)


def random_state_batch_doubled_impl(hilb: TensorHilbert, key, batches, dtype):
    shape = (batches, hilb.size)

    keys = jax.random.split(key, hilb._n_hilbert_spaces)

    vs = [
        random_state_batch(hi, k, batches, dtype)
        for (hi, k) in zip(hilb._hilbert_spaces, keys)
    ]

    return jnp.concatenate(vs, axis=1)


def _make_subfun(hilb, i, sub_hi):
    def subfun(args):
        key, state, index = args

        # jax.experimental.host_callback.id_print(index, text=f"printing subfun_{i}:")

        sub_state = state[hilb._cum_indices[i] : hilb._cum_sizes[i]]
        new_sub_state, old_val = flip_state_scalar(
            sub_hi, key, sub_state, index - hilb._cum_indices[i]
        )
        idx = jax.ops.index[hilb._cum_indices[i] : hilb._cum_sizes[i]]
        new_state = jax.ops.index_update(state, idx, new_sub_state)
        return new_state, old_val

    return subfun


## flips
from jax import experimental
from jax.experimental import host_callback


def flip_state_scalar_doubled(hilb: TensorHilbert, key, state, index):

    subfuns = []
    for (i, sub_hi) in enumerate(hilb._hilbert_spaces):
        subfuns.append(_make_subfun(hilb, i, sub_hi))

    branches = []
    for i in hilb._hilbert_i:
        branches.append(subfuns[i])

    return jax.lax.switch(index, branches, (key, state, index))


register_random_state_impl(TensorHilbert, batch=random_state_batch_doubled_impl)
register_flip_state_impl(TensorHilbert, scalar=flip_state_scalar_doubled)
