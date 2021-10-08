import jax
from functools import partial


def _treeify(f):
    def _f(x, *args, **kwargs):
        return jax.tree_map(lambda y: f(y, *args, **kwargs), x)

    return _f


@_treeify
def _unbatch(x):
    return x.reshape((-1,) + x.shape[2:])


@_treeify
def _batch(x, batch_size=None):
    # batch_size=None -> add just a dummy batch dimension, same as np.expand_dims(x, 0)
    n = x.shape[0]
    if batch_size is None:
        batch_size = n

    n_batches, residual = divmod(n, batch_size)
    if residual != 0:
        raise ValueError("the first dimension of x must be divisible by batch_size")
    return x.reshape((n_batches, batch_size) + x.shape[1:])


def _batch_size(x):
    b = set(map(lambda x: x.shape[:2], jax.tree_leaves(x)))
    if len(b) != 1:
        raise ValueError(
            "The arrays in x have inconsistent batch_size or number of batches"
        )
    return b.pop()[1]


def unbatch(x_batched):
    """
    Merge the first two axes of an array (or a pytree of arrays)
    Args:
        x_batched: an array (or pytree of arrays) of at least 2 dimensions
    Returns: a pair (x, batch_fn)
        where x is x_batched reshaped to (-1,)+x.shape[2:]
        and batch_fn is a function which restores x given x_batched
    """
    return _unbatch(x_batched), partial(_batch, batch_size=_batch_size(x_batched))


def batch(x, batch_size=None):
    """
    Split an array (or a pytree of arrays) into batches along the first axis
    Args:
        x: an array (or pytree of arrays)
        batch_size: an integer or None (default)
            The first axis in x must be a multiple of batch_size
    Returns: a pair (x_batched, unbatch_fn) where
        - x_batched is x reshaped to (-1, batch_size)+x.shape[1:]
          if batch_size is None then it defaults to x.shape[0], i.e. just one batch
        - unbatch_fn is a function which restores x given x_batched
    """
    return _batch(x, batch_size), _unbatch
