from typing import Optional, List

import jax
import numpy as np
from jax import numpy as jnp

# from numba import jit

from netket.hilbert import DoubledHilbert

from .base import (
    random_state_batch,
    random_state_batch_impl,
    flip_state_scalar,
    flip_state_scalar_impl,
)


@random_state_batch_impl.register
def random_state_batch_doubled_impl(hilb: DoubledHilbert, key, batches, dtype):
    shape = (batches, hilb.size)

    key1, key2 = jax.random.split(key)

    v1 = random_state_batch(hilb.physical, key1, batches, dtype)
    v2 = random_state_batch(hilb.physical, key2, batches, dtype)

    return jnp.concatenate([v1, v2], axis=1)


## flips
@flip_state_scalar_impl.register
def flip_state_scalar_doubled(hilb: DoubledHilbert, key, state, index):
    def flip_lower_state_scalar(args):
        key, state, index = args
        return flip_state_scalar(hilb.physical, key, state, index)

    def flip_upper_state_scalar(args):
        key, state, index = args
        return flip_state_scalar(hilb.physical, key, state, index - hilb.size)

    return jax.lax.cond(
        index < hilb.physical.size,
        flip_lower_state_scalar,
        flip_upper_state_scalar,
        (key, state, index),
    )
