from typing import Optional, List

import jax
import numpy as np
from jax import numpy as jnp

# from numba import jit

from netket.hilbert import Fock

from .base import flip_state_scalar_impl


## flips
@flip_state_scalar_impl.register
def flip_state_scalarspin(hilb: Fock, key, σ, idx):
    n_states = hilb._n_max + 1

    σi_old = σ[idx]
    r = jax.random.uniform(key)
    σi_new = jax.numpy.floor(r * (n_states - 1))
    σi_new = σi_new + (σi_new >= σi_old)

    return jax.ops.index_update(σ, idx, σi_new), σi_old
