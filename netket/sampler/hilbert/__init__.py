"""
This module constains functions that act on hilbert spaces necessary
for sampling.

This module exports two functions: 
 - `random_state(hilbert, key, batch_size, dtype)` which generates a batch of random
 states in the given hilbert space, in an array with the specified dtype. 
 - `flip_state(hilb, key, states, indices)` which, for every state σ in the batch of states,
 considers σᵢ and returns a new state where that entry is different from the previous.
 The new configuration is selected with uniform probability among the local possible
 configurations.

Those methods can be implemented both as scalar functions by overloading
`xxx_scalar_impl` or as batched functions by overloading `xxx_batch_impl`.
If only the scalar is defined, then jax.vmap is used to map over the batch
axis. Of course, it is usually more performant to define the batched code, 
but that is not always possible.

Each one of those functions has 
"""
from .base import flip_state

from . import custom
from . import qubit
from . import spin
from . import fock
from . import doubled
