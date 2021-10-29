import jax
from functools import partial


def _treeify(f):
    def _f(x, *args, **kwargs):
        return jax.tree_map(lambda y: f(y, *args, **kwargs), x)

    return _f


@_treeify
def _unchunk(x):
    return x.reshape((-1,) + x.shape[2:])


@_treeify
def _chunk(x, chunk_size=None):
    # chunk_size=None -> add just a dummy chunk dimension, same as np.expand_dims(x, 0)
    n = x.shape[0]
    if chunk_size is None:
        chunk_size = n

    n_chunks, residual = divmod(n, chunk_size)
    if residual != 0:
        raise ValueError(
            "The first dimension of x must be divisible by chunk_size."
            + f"\n            Got x.shape={x.shape} but chunk_size={chunk_size}."
        )
    return x.reshape((n_chunks, chunk_size) + x.shape[1:])


def _chunk_size(x):
    b = set(map(lambda x: x.shape[:2], jax.tree_leaves(x)))
    if len(b) != 1:
        raise ValueError(
            "The arrays in x have inconsistent chunk_size or number of chunks"
        )
    return b.pop()[1]


def unchunk(x_chunked):
    """
    Merge the first two axes of an array (or a pytree of arrays)
    Args:
        x_chunked: an array (or pytree of arrays) of at least 2 dimensions
    Returns: a pair (x, chunk_fn)
        where x is x_chunked reshaped to (-1,)+x.shape[2:]
        and chunk_fn is a function which restores x given x_chunked
    """
    return _unchunk(x_chunked), partial(_chunk, chunk_size=_chunk_size(x_chunked))


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
    return _chunk(x, chunk_size), _unchunk
