from typing import Optional, List

import jax
import numpy as np
from jax import numpy as jnp

# from numba import jit

from netket.hilbert import CustomHilbert

from .base import flip_state_batch_impl


## flips
@flip_state_batch_impl.register
def flip_state_batch_spin(hilb: CustomHilbert, key, σ, indxs):
    n_batches = σ.shape[0]

    local_states = jnp.asarray(hilb.local_states)

    rs = jax.random.randint(
        key, shape=(n_batches,), minval=0, maxval=len(hilb.local_states) - 1
    )

    def scalar_update_fun(σ, indx, rs):
        new_val = local_states[rs + (local_states[rs] >= σ[indx])]
        return jax.ops.index_update(σ, indx, new_val), σ[indx]

    return jax.vmap(scalar_update_fun, in_axes=(0, 0, 0), out_axes=0)(σ, indxs, rs)
