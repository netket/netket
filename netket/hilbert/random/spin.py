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
from functools import partial

from netket.hilbert import Spin
from netket.utils.dispatch import dispatch

from .fock import _choice


@dispatch
def random_state(hilb: Spin, key, batches: int, *, dtype=np.float32):
    shape = (batches,)
    if hilb._total_sz is None:
        return _random_states(hilb, key, shape, dtype)
    else:
        return _random_states_with_constraint(hilb, key, shape, dtype)


@partial(jax.jit, static_argnames=("hilb", "shape", "dtype"))
def _random_states(hilb, key, shape, dtype):
    S = hilb._s
    n_states = int(2 * S + 1)
    rs = jax.random.randint(key, shape=shape + (hilb.size,), minval=0, maxval=n_states)
    return (2 * rs - (n_states - 1)).astype(dtype)


@partial(jax.jit, static_argnames=("hilb", "shape", "dtype"))
def _random_states_with_constraint(hilb, rngkey, shape, dtype):
    # Generate random spin states with a given hilb._total_sz.
    # Note that this is NOT a uniform distribution over the
    # basis states of the constrained hilbert space.

    N = hilb.size
    S = hilb._s
    n_states = int(2 * S) + 1
    # if constrained and S == 1/2, use a trick to sample quickly
    if n_states == 2:
        m = int(hilb._total_sz * 2)
        nup = (N + m) // 2
        ndown = (N - m) // 2
        x = jnp.ones(shape + (nup + ndown,), dtype=dtype).at[..., -ndown:].set(-1)
        return jax.random.permutation(rngkey, x, axis=-1, independent=True)
    else:
        # if constrained and S != 1/2, then use a slow fallback algorithm
        # TODO: find better, faster way to sample constrained arbitrary spaces.

        # start with all spins in the state with the lowest eigenvalue
        init = jnp.full(shape + (hilb.size,), -round(2 * hilb._s), dtype=dtype)

        def body_fn(out, key):
            # find all spins which are not yet in the highest state
            mask = out <= round(2 * hilb._s - 1)
            # select one of those spins uniformly and change it's state to the next higher one
            out = jax.lax.select(_choice(key, mask), out + 2, out)
            return out, None

        # iterate until total_sz is reached
        n = round(hilb._s * hilb.size + hilb._total_sz)
        keys = jax.random.split(rngkey, n)
        return jax.lax.scan(body_fn, init, keys)[0]


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
    xi_new = xi_new.astype(x.dtype)

    new_state = x.at[i].set(xi_new)
    return new_state, xi_old
