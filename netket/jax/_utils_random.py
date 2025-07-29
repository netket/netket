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
from jax.sharding import NamedSharding, PartitionSpec as P

from netket.utils import random_seed, config
from netket.utils.types import PRNGKeyT, SeedT


def PRNGKey(seed: SeedT | None = None, *, root: int = 0) -> PRNGKeyT:
    """
    Initialises a PRNGKey using an optional starting seed.

    If using sharding, the returned key will be replicated on every process.

    Args:
        seed: An optional integer value to use as seed
        root: the master rank, used when running with multiple nodes (default 0)

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

        key = jax.random.key(seed)
    elif isinstance(seed, jax.Array):
        if jnp.issubdtype(seed.dtype, jax.dtypes.prng_key):
            # new-style keys
            key = seed
        elif jnp.issubdtype(seed.dtype, jax.numpy.unsignedinteger):
            # old-style PRNGKey
            key = seed
        else:
            raise TypeError(
                f"Expected seed to be an integer or a PRNGKey, got {type(seed)}, {seed.dtype} : {seed}"
            )
    else:
        raise TypeError(f"unsupported type {type(seed)}")

    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty:
        if len(mesh.explicit_axes) >= 0:
            key = jax.sharding.reshard(key, P())
        else:
            sharding = NamedSharding(mesh, P())
            key = jax.lax.with_sharding_constraint(key, sharding)
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


def _bcast_key(key, root=0) -> PRNGKeyT:
    """
    Utility function equivalent to broadcast a random key to multiple jax processes
    and make sure it is the same everywhere.
    """
    is_new_style_key = jnp.issubdtype(key.dtype, jax.dtypes.prng_key)

    if is_new_style_key:
        _impl = jax.random.key_impl(key)
        key = jax.random.key_data(key)

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
