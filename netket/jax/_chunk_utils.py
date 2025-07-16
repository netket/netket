from functools import partial

import jax
import jax.numpy as jnp
from netket.utils.iterators import safe_map

safe_zip = partial(zip, strict=True)


def _tree_map_multi_out(f, tree, *rest, is_leaf=None):
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    res = [f(*xs) for xs in safe_zip(*all_leaves)]
    return tuple(safe_map(treedef.unflatten, safe_zip(*res)))


def _treeify(f):
    def _f(x, *args, **kwargs):
        return _tree_map_multi_out(lambda y: f(y, *args, **kwargs), x)

    return _f


def _unchunk(x, _x_rest=None):
    res = jax.tree_util.tree_map(
        partial(jax.lax.collapse, start_dimension=0, stop_dimension=2), x
    )
    if _x_rest is not None:
        return jax.tree_util.tree_map(
            lambda *a: jnp.concatenate(a, axis=0), res, _x_rest
        )
    else:
        return res


@_treeify
def _chunk(x, chunk_size=None):
    # chunk_size=None -> add just a dummy chunk dimension, same as np.expand_dims(x, 0)
    if x.ndim == 0:
        raise ValueError("x cannot be chunked as it has 0 dimensions.")
    n = x.shape[0]
    if chunk_size is None:
        chunk_size = n

    n_chunks, n_rest = divmod(n, chunk_size)
    x_chunks = x[: n - n_rest, ...].reshape((n_chunks, chunk_size) + x.shape[1:])
    x_rest = x[n - n_rest :, ...]
    return x_chunks, x_rest


def _chunk_size(x):
    b = set(map(lambda x: x.shape[:2], jax.tree_util.tree_leaves(x)))
    if len(b) != 1:
        raise ValueError(
            "The arrays in x have inconsistent chunk_size or number of chunks"
        )
    return b.pop()[1]


def _chunk_without_rest(x, chunk_size):
    x_chunks, x_rest = _chunk(x, chunk_size)

    # error if we have rest
    def _check(x, x_rest):
        if x_rest.size != 0:
            raise ValueError(
                "The first dimension of x must be divisible by chunk_size."
                + f"\nGot x.shape={x.shape} but chunk_size={chunk_size}."
            )
        return x

    jax.tree.map(_check, x, x_rest)
    return x_chunks


def unchunk(x_chunked):
    """
    Merge the first two axes of an array (or a pytree of arrays)
    Args:
        x_chunked: an array (or pytree of arrays) of at least 2 dimensions
    Returns: a pair (x, chunk_fn)
        where x is x_chunked reshaped to (-1,)+x.shape[2:]
        and chunk_fn is a function which restores x given x_chunked
    """
    return _unchunk(x_chunked), partial(
        _chunk_without_rest, chunk_size=_chunk_size(x_chunked)
    )


def chunk(x, chunk_size=None):
    """
    Split an array (or a pytree of arrays) into chunks along the first axis

    Args:
        x: an array (or pytree of arrays)
        chunk_size: an integer or None (default)
            The first axis in x must be a multiple of chunk_size
    Returns: a pair (x_chunked, unchunk_fn) where
        - x_chunked is x reshaped to (-1, chunk_size)+x.shape[1:]
          if chunk_size is None then it defaults to x.shape[0], i.e. just one chunk
        - unchunk_fn is a function which restores x given x_chunked
    """

    return _chunk_without_rest(x, chunk_size), _unchunk
