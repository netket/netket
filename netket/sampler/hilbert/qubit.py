import jax
from jax import numpy as jnp

from netket.hilbert import Qubit

from .base import random_state_batch_impl, flip_state_scalar_impl


@random_state_batch_impl.register
def jax_random_state_batch_qubit_impl(hilb: Qubit, key, batches, dtype):
    shape = (batches, hilb.size)

    rs = jax.random.randint(key, shape=shape, minval=0, maxval=2)
    return jnp.asarray(rs, dtype=dtype)


## flips
@flip_state_scalar_impl.register
def flip_state_scalar_spin(hilb: Qubit, key, state, index):
    return jax.ops.index_update(x, i, -x[i] + 1), x[i]
