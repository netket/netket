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


import jax
import jax.numpy as jnp

from netket.utils import random_seed, mpi, config
from netket.utils.mpi import MPI_jax_comm
from netket.utils.types import PRNGKeyT, SeedT


def PRNGKey(seed: SeedT | None = None, *, root: int = 0, comm=MPI_jax_comm) -> PRNGKeyT:
    """
    Initialises a PRNGKey using an optional starting seed.

    If using sharding, the returned key will be replicated while if using MPI
    the key of the master rank will be broadcasted to every process.

    Args:
        seed: An optional integer value to use as seed
        root: the master rank, used when running under MPI (defaults to 0)
        comm: The MPI communicator to use for broadcasting, if necessary

    Returns:
        A sharded/broadcasted :func:`jax.random.PRNGKey`.

    """
    if seed is None:
        seed = random_seed()

    if isinstance(seed, int):
        # We can't sync the PRNGKey, so we can only sinc integer seeds
        # see https://github.com/google/jax/pull/16511
        if config.netket_experimental_sharding and jax.process_count() > 1:  # type: ignore[attr-defined]
            # TODO: use stable jax function
            from jax.experimental import multihost_utils

            seed = int(
                multihost_utils.broadcast_one_to_all(
                    seed, is_source=jax.process_index() == root
                ).item()
            )

        key = jax.random.PRNGKey(seed)
    else:
        key = seed

    if config.netket_experimental_sharding:
        key = jax.lax.with_sharding_constraint(
            key, jax.sharding.PositionalSharding(jax.devices()).replicate()
        )
    else:  # type: ignore[attr-defined]
        key = _bcast_key(key, root=root, comm=comm)
    return key


class PRNGSeq:
    """
    A sequence of PRNG keys generated based on an initial key.
    """

    def __init__(self, base_key: SeedT | None = None):
        if base_key is None:
            base_key = PRNGKey()
        elif isinstance(base_key, int):
            base_key = PRNGKey(base_key)
        self._current: PRNGKeyT = base_key

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

    keys = _bcast_key(keys, root=root, comm=comm)

    return keys[mpi.rank]


def _bcast_key(key, root=0, comm=MPI_jax_comm) -> PRNGKeyT:
    """
    Utility function equivalent to calling `mpi_bcast_jax` on a jax key,
    but working around some sharding bug when not using sharding, arising
    from MPI.
    """
    is_new_style_key = jnp.issubdtype(key.dtype, jax.dtypes.prng_key)

    if is_new_style_key:
        _impl = jax.random.key_impl(key)
        key = jax.random.key_data(key)

    key = jax.tree_util.tree_map(
        lambda k: mpi.mpi_bcast_jax(k, root=root, comm=comm)[0], key
    )

    if is_new_style_key:
        key = jax.random.wrap_key_data(key, impl=_impl)  # type: ignore[arg-type]

    return key


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
