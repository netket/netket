# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional


import jax

from netket.utils import random_seed, mpi
from netket.utils.mpi import MPI_jax_comm
from netket.utils.types import PRNGKeyT, SeedT


def PRNGKey(
    seed: Optional[SeedT] = None, *, root: int = 0, comm=MPI_jax_comm
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

    key = jax.tree_util.tree_map(
        lambda k: mpi.mpi_bcast_jax(k, root=root, comm=comm)[0], key
    )

    return key


class PRNGSeq:
    """
    A sequence of PRNG keys generated based on an initial key.
    """

    def __init__(self, base_key: Optional[SeedT] = None):
        if base_key is None:
            base_key = PRNGKey()
        elif isinstance(base_key, int):
            base_key = PRNGKey(base_key)
        self._current = base_key

    def __iter__(self):
        return self

    def __next__(self):
        self._current = jax.random.split(self._current, num=1)[0]
        return self._current

    def next(self):
        return self.__next__()

    def take(self, num: int):
        """
        Returns an array of `num` PRNG keys and advances the iterator accordingly.
        """
        keys = jax.random.split(self._current, num=num + 1)
        self._current = keys[-1]
        return keys[:-1]


def mpi_split(key, *, root=0, comm=MPI_jax_comm) -> PRNGKeyT:
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

    keys = jax.tree_util.tree_map(lambda k: mpi.mpi_bcast_jax(k, root=root)[0], keys)

    return keys[mpi.rank]


def batch_choice(key, a, p):
    """
    Batched version of `jax.random.choice`.

    Attributes:
      key: a PRNGKey used as the random key.
      a: 1D array. Random samples are generated from its elements.
      p: 2D array of shape `(batch_size, a.size)`. Each slice `p[i, :]` is
        the probabilities associated with entries in `a` to generate a sample
        at the index `i` of the output. Can be unnormalized.

    Returns:
      The generated samples as an 1D array of shape `(batch_size,)`.
    """
    p_cumsum = p.cumsum(axis=1)
    r = p_cumsum[:, -1:] * jax.random.uniform(key, shape=(p.shape[0], 1))
    indices = (r > p_cumsum).sum(axis=1)
    out = a[indices]
    return out
