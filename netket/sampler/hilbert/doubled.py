from typing import Optional, List

import jax
import numpy as np
from jax import numpy as jnp

# from numba import jit

from netket.hilbert import DoubledHilbert

from .base import (
    flip_state_scalar,
    flip_state_scalar_impl,
)


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
