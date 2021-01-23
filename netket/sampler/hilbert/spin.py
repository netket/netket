from typing import Optional, List

import jax
import numpy as np
from jax import numpy as jnp

# from numba import jit

from netket.hilbert import Spin

from .base import flip_state_scalar_impl


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
