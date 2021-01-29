import jax
from jax import numpy as jnp

from netket.hilbert import Qubit

from .base import flip_state_scalar_impl, random_state_batch_impl


@random_state_batch_impl.register
def _random_state_batch_impl(hilb: Qubit, key, batches, dtype):
    rs = jax.random.randint(key, shape=(batches, hilb.size), minval=0, maxval=2)
    return jnp.asarray(rs, dtype=dtype)


## flips
@flip_state_scalar_impl.register
def flip_state_scalar_spin(hilb: Qubit, key, state, index):
    return jax.ops.index_update(x, i, -x[i] + 1), x[i]
