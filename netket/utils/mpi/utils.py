from typing import Optional
import jax

from netket.utils.types import SeedT, PRNGKeyT

from ..seed import random_seed

from . import primitives
from . import mpi

def split_key(key, *, root=0, comm=mpi.MPI_jax_comm) -> PRNGKeyT:
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

    # Maybe add error/warning if in_key is not the same
    # on all MPI nodes?
    keys = jax.random.split(key, mpi.n_nodes)

    keys = jax.tree_map(lambda k: primitives.mpi_bcast_jax(k, root=root)[0], keys)

    return keys[mpi.rank]

def PRNGKey(
    seed: Optional[SeedT] = None, *, root: int = 0, comm=mpi.MPI_jax_comm
) -> PRNGKeyT:
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

    key = jax.tree_map(lambda k: primitives.mpi_bcast_jax(k, root=root, comm=comm)[0], key)
    return key
