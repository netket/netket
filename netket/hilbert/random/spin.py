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
import numpy as np
from jax import numpy as jnp

from netket.hilbert import Spin
from netket.utils.dispatch import dispatch


@dispatch
def random_state(hilb: Spin, key, batches: int, *, dtype=np.float32):
    S = hilb._s
    shape = (batches, hilb.size)

    # If unconstrained space, use fast sampling
    if hilb._total_sz is None:
        n_states = int(2 * S + 1)
        rs = jax.random.randint(key, shape=shape, minval=0, maxval=n_states)

        two = jnp.asarray(2, dtype=dtype)
        return jnp.asarray(rs * two - (n_states - 1), dtype=dtype)
    else:
        N = hilb.size
        n_states = int(2 * S) + 1
        # if constrained and S == 1/2, use a trick to sample quickly
        if n_states == 2:
            m = int(hilb._total_sz * 2)
            nup = (N + m) // 2
            ndown = (N - m) // 2

            x = jnp.concatenate(
                (
                    jnp.ones((batches, nup), dtype=dtype),
                    -jnp.ones(
                        (
                            batches,
                            ndown,
                        ),
                        dtype=dtype,
                    ),
                ),
                axis=1,
            )

            # deprecated: return jax.random.shuffle(key, x, axis=1)
            return jax.vmap(jax.random.permutation)(
                jax.random.split(key, x.shape[0]), x
            )

        # if constrained and S != 1/2, then use a slow fallback algorithm
        # TODO: find better, faster way to smaple constrained arbitrary spaces.
        else:
            from jax.experimental import host_callback as hcb

            state = hcb.call(
                lambda rng: _random_states_with_constraint(hilb, rng, batches, dtype),
                key,
                result_shape=jax.ShapeDtypeStruct(shape, dtype),
            )

            return state


# TODO: could numba-jit this
def _random_states_with_constraint(hilb, rngkey, n_batches, dtype):
    out = np.full((n_batches, hilb.size), -round(2 * hilb._s), dtype=dtype)
    rgen = np.random.default_rng(rngkey)

    for b in range(n_batches):
        sites = list(range(hilb.size))
        ss = hilb.size

        for i in range(round(hilb._s * hilb.size + hilb._total_sz)):
            s = rgen.integers(0, ss, size=())

            out[b, sites[s]] += 2

            if out[b, sites[s]] > round(2 * hilb._s - 1):
                sites.pop(s)
                ss -= 1

    return out


@dispatch
def flip_state_scalar(hilb: Spin, key, state, index):
    if hilb._s == 0.5:
        return _flipat_N2(key, state, index)
    else:
        return _flipat_generic(key, state, index, hilb._s)


def _flipat_N2(key, x, i):
    res = x.at[i].set(-x[i]), x[i]
    return res


def _flipat_generic(key, x, i, s):
    n_states = int(2 * s + 1)

    xi_old = x[i]
    r = jax.random.uniform(key)
    xi_new = jax.numpy.floor(r * (n_states - 1)) * 2 - (n_states - 1)
    xi_new = xi_new + 2 * (xi_new >= xi_old)

    new_state = x.at[i].set(xi_new)
    return new_state, xi_old
