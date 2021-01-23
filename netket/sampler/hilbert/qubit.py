import jax
from jax import numpy as jnp

from netket.hilbert import Qubit

from .base import flip_state_scalar_impl


## flips
@flip_state_scalar_impl.register
def flip_state_scalar_spin(hilb: Qubit, key, state, index):
    return jax.ops.index_update(x, i, -x[i] + 1), x[i]
