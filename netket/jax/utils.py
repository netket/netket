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

from functools import partial
from typing import Optional, Tuple, Any, Union, Tuple, Callable

import numpy as np

import jax
import netket.jax as nkjax
from jax import numpy as jnp
from jax.tree_util import (
    tree_flatten,
    tree_unflatten,
    tree_map,
    tree_multimap,
    tree_leaves,
)
from jax.util import as_hashable_function
from jax.dtypes import dtype_real

from netket.utils import MPI, n_nodes, rank, random_seed
from netket.utils.types import PyTree, PRNGKeyT, SeedT


def tree_ravel(pytree: PyTree) -> Tuple[jnp.ndarray, Callable]:
    """Ravel (i.e. flatten) a pytree of arrays down to a 1D array.

    Args:
      pytree: a pytree to ravel

    Returns:
      A pair where the first element is a 1D array representing the flattened and
      concatenated leaf values, and the second element is a callable for
      unflattening a 1D vector of the same length back to a pytree of of the same
      structure as the input ``pytree``.
    """
    leaves, treedef = tree_flatten(pytree)
    flat, unravel_list = nkjax.vjp(_ravel_list, *leaves)
    unravel_pytree = lambda flat: tree_unflatten(treedef, unravel_list(flat))
    return flat, unravel_pytree


def _ravel_list(*lst):
    return jnp.concatenate([jnp.ravel(elt) for elt in lst]) if lst else jnp.array([])


def eval_shape(fun, *args, has_aux=False, **kwargs):
    """
    Returns the dtype of forward_fn(pars, v)
    """
    if has_aux:
        out, _ = jax.eval_shape(fun, *args, **kwargs)
    else:
        out = jax.eval_shape(fun, *args, **kwargs)
    return out


def tree_size(tree: PyTree) -> int:
    """
    Returns the sum of the size of all leaves in the tree.
    It's equivalent to the number of scalars in the pytree.
    """
    return sum(tree_leaves(tree_map(lambda x: x.size, tree)))


def is_complex(x):
    #  Returns true if x is complex
    return jnp.issubdtype(x.dtype, jnp.complexfloating)


def tree_leaf_iscomplex(pars):
    """
    Returns true if at least one leaf in the tree has complex dtype.
    """
    return any(jax.tree_leaves(jax.tree_map(is_complex, pars)))


def is_complex_dtype(typ):
    return jnp.issubdtype(typ, jnp.complexfloating)


def dtype_complex(typ):
    """
    Return the complex dtype corresponding to the type passed in.
    If it is already complex, do nothing
    """
    if is_complex_dtype(typ):
        return typ
    elif typ == np.dtype("float32"):
        return np.dtype("complex64")
    elif typ == np.dtype("float64"):
        return np.dtype("complex128")
    else:
        raise TypeError("Unknown complex type for {}".format(typ))


def maybe_promote_to_complex(*types):
    """
    Maybe promotes the first argument to it's complex counterpart given by
    dtype_complex(typ) if any of the arguments is complex
    """
    main_typ = types[0]

    for typ in types:
        if is_complex_dtype(typ):
            return dtype_complex(main_typ)
    else:
        return main_typ


class HashablePartial(partial):
    """
    A class behaving like functools.partial, but that retains it's hash
    if it's created with a lexically equivalent (the same) function and
    with the same partially applied arguments and keywords.

    It also stores the computed hash for faster hashing.
    """

    def __init__(self, *args, **kwargs):
        self._hash = None

    def __eq__(self, other):
        return (
            type(other) is HashablePartial
            and self.func.__code__ == other.func.__code__
            and self.args == other.args
            and self.keywords == other.keywords
        )

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(
                (self.func.__code__, self.args, frozenset(self.keywords.items()))
            )

        return self._hash

    def __repr__(self):
        return f"<hashable partial {self.func.__name__} with args={self.args} and kwargs={self.keywords}, hash={hash(self)}>"


# jax.tree_util.register_pytree_node(
#    HashablePartial,
#    lambda partial_: ((), (partial_.func, partial_.args, partial_.keywords)),
#    lambda args, _: StaticPartial(args[0], *args[1], **args[2]),
# )


def PRNGKey(
    seed: Optional[SeedT] = None, root: int = 0, comm=MPI.COMM_WORLD
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

    if n_nodes > 1:
        import mpi4jax

        key, _ = mpi4jax.bcast(key, root=root, comm=comm)

    return key


def mpi_split(key, root=0, comm=MPI.COMM_WORLD) -> PRNGKeyT:
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
    keys = jax.random.split(key, n_nodes)

    if n_nodes > 1:
        import mpi4jax

        keys, _ = mpi4jax.bcast(keys, root=root, comm=comm)

    return keys[rank]


class PRNGSeq:
    """
    A sequence of PRNG keys genrated based on an initial key.
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
