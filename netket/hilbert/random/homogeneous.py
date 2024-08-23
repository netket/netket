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

from functools import partial
import warnings

import jax
import jax.numpy as jnp

from netket.errors import UnoptimisedCustomConstraintRandomStateMethodWarning
from netket.hilbert import HomogeneousHilbert
from netket.utils.dispatch import dispatch


@dispatch
def random_state(hilb: HomogeneousHilbert, key, batches: int, *, dtype=None):
    return random_state(hilb, hilb.constraint, key, batches, dtype=dtype)


@dispatch
@partial(jax.jit, static_argnames=("hilb", "batches", "dtype"))
def random_state(  # noqa: F811
    hilb: HomogeneousHilbert, constraint: None, key, batches: int, *, dtype=None
):
    if dtype is None:
        dtype = hilb._local_states.dtype

    x_ids = jax.random.randint(
        key, shape=(batches, hilb.size), minval=0, maxval=len(hilb._local_states)
    )
    return hilb.local_indices_to_states(x_ids)


@dispatch
def random_state(  # noqa: F811
    hilb: HomogeneousHilbert, constraint, key, batches: int, *, dtype=None
):
    warnings.warn(UnoptimisedCustomConstraintRandomStateMethodWarning(hilb, constraint))

    keys = jax.random.split(key, batches + 1)
    states = random_state(hilb, None, keys[0], batches, dtype=dtype)

    def _loop_until_ok(state, key):
        def __body(args):
            state, _key = args
            _key, subkey = jax.random.split(_key)
            new_state = random_state(hilb, None, subkey, 1, dtype=dtype)[0]
            return (new_state, _key)

        def __cond(args):
            state, _ = args
            return jnp.logical_not(constraint(state))

        return jax.lax.while_loop(__cond, __body, (state, key))[0]

    return jax.vmap(_loop_until_ok, in_axes=(0, 0))(states, keys[1:])
