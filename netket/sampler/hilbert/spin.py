from typing import Optional, List

import jax
import numpy as np
from jax import numpy as jnp

# from numba import jit

from netket.hilbert import Spin

from .base import random_state_batch_impl, flip_state_scalar_impl


@random_state_batch_impl.register
def random_state_batch_impl_spin(hilb: Spin, key, batches, dtype):
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
            m = hilb._total_sz * 2
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

            cb = lambda rng: _random_states_with_constraint(hilb, rng, batches, dtype)

            state = hcb.call(
                cb,
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

        for i in range(round(hilb._s * hilb.size) + hilb._total_sz):
            s = rgen.integers(0, ss, size=())

            out[b, sites[s]] += 2

            if out[b, sites[s]] > round(2 * hilb._s - 1):
                sites.pop(s)
                ss -= 1

    return out


## flips
@flip_state_scalar_impl.register
def flip_state_scalar_spin(hilb: Spin, key, state, index):
    if hilb._s == 0.5:
        return _flipat_N2(key, state, index)
    else:
        return _flipat_generic(key, state, index, hilb._s)


def _flipat_N2(key, x, i):
    return jax.ops.index_update(x, i, -x[i]), x[i]


def _flipat_generic(key, x, i, s):
    n_states = int(2 * s + 1)

    xi_old = x[i]
    r = jax.random.uniform(key)
    xi_new = jax.numpy.floor(r * (n_states - 1)) * 2 - (n_states - 1)
    xi_new = xi_new + 2 * (xi_new >= xi_old)

    return jax.ops.index_update(x, i, xi_new), xi_old
