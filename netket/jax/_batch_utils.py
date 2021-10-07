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


def unbatch(x):
    return _unbatch(x), partial(_batch, batch_size=_batch_size(x))


def batch(x, batch_size=None):
    return _batch(x, batch_size), _unbatch
