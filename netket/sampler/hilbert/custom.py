from typing import Optional, List

import jax
import numpy as np
from jax import numpy as jnp

# from numba import jit

from netket.hilbert import CustomHilbert

from .base import random_state_batch_impl, flip_state_batch_impl


@random_state_batch_impl.register
def random_state_batch_impl_spin(hilb: CustomHilbert, key, batches, dtype):
    if not hilb.is_discrete or not hilb.is_finite or hilb._has_constraint:
        raise NotImplementedError()

    # Default version for discrete hilbert spaces without constraints.
    # More specialized initializations can be defined in the derived classes.

    shape = (batches, hilb._size)

    σ = jax.random.choice(key, self.local_states, shape=shape, replace=True)
    return jnp.asarray(σ, dtype=dtype)


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
