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

from netket.hilbert import Fock

from .base import register_flip_state_impl, register_random_state_impl


def random_state_batch_fock_impl(hilb: Fock, key, batches, dtype):
    shape = (batches, hilb.size)

    # If unconstrained space, use fast sampling
    if hilb.n_particles is None:
        rs = jax.random.randint(key, shape=shape, minval=0, maxval=hilb.n_max + 1)
        return jnp.asarray(rs, dtype=dtype)

    else:
        from jax.experimental import host_callback as hcb

        cb = lambda rng: _random_states_with_constraint(hilb, rng, batches, dtype)

        state = hcb.call(
            cb,
            key,
            result_shape=jax.ShapeDtypeStruct(shape, dtype),
        )

        return state


def _random_states_with_constraint(hilb, rngkey, n_batches, dtype):
    out = np.zeros((n_batches, hilb.size), dtype=dtype)
    rgen = np.random.default_rng(rngkey)

    for b in range(n_batches):
        sites = list(range(hilb.size))
        ss = hilb.size

        for i in range(hilb.n_particles):
            s = rgen.integers(0, ss, size=())

            out[b, sites[s]] += 1

            if out[b, sites[s]] == hilb.n_max:
                sites.pop(s)
                ss -= 1

    return out


## flips
def flip_state_scalar_fock(hilb: Fock, key, σ, idx):
    if hilb._n_max == 0:
        return σ, σ[idx]

    n_states = hilb._n_max + 1

    σi_old = σ[idx]
    r = jax.random.uniform(key)
    σi_new = jax.numpy.floor(r * (n_states - 1))
    σi_new = σi_new + (σi_new >= σi_old)

    return jax.ops.index_update(σ, idx, σi_new), σi_old


register_random_state_impl(Fock, batch=random_state_batch_fock_impl)
register_flip_state_impl(Fock, scalar=flip_state_scalar_fock)
