import jax
from functools import partial

import numpy as np
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


def eval_shape(fun, *args, has_aux=False, **kwargs):
    """
    Returns the dtype of forward_fn(pars, v)
    """
    if has_aux:
        out, _ = jax.eval_shape(fun, *args, **kwargs)
    else:
        out = jax.eval_shape(fun, *args, **kwargs)
    return out


def tree_size(tree):
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


def dtype_is_complex(typ):
    return jnp.issubdtype(typ, jnp.complexfloating)


def dtype_complex(typ):
    """
    Return the complex dtype corresponding to the type passed in.
    If it is already complex, do nothing
    """
    if dtype_is_complex(typ):
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
        if dtype_is_complex(typ):
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
