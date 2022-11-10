from functools import partial
from typing import Optional

import jax

from flax.jax_utils import replicate

from netket.utils.types import SeedT, PRNGKeyT
from ..seed import random_seed

@partial(jax.pmap, axis_name = "mpi")
def split_key(key) -> PRNGKeyT:
    """
    Split a key across MPI nodes in the communicator.
    Only the input key on the root process matters.

    Arguments:
        key: The key to split. Only considered the one on the root process.
        root: (default=0) The root rank from which to take the input key.
        comm: (default=MPI.COMM_WORLD) The MPI communicator.

    Returns:
        A PRNGKey depending on rank number and key.
    """
    return jax.random.fold_in(key, jax.lax.axis_index('mpi'))



def PRNGKey(
    seed: Optional[SeedT] = None) -> PRNGKeyT:
    """
    Initialises a PRNGKey using an optional starting seed.
    The same seed will be distributed to all processes.
    """
    if seed is None:
        key = jax.random.PRNGKey(random_seed())
    elif isinstance(seed, int):
        key = jax.random.PRNGKey(seed)
    else:
        key = seed

    return replicate(key)
