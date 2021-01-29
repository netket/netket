import jax

from jax import tree_map as _tree_map, tree_multimap, tree_flatten, tree_unflatten
import jax.numpy as jnp


def info(obj, depth=None):
    if hasattr(obj, "info"):
        return obj.info(depth)
    else:
        return str(obj)


def tree_map(fun, tree, *args, **kwargs):
    return _tree_map(lambda x: fun(x, *args, **kwargs), tree)


def trees2_map(fun, tree1, tree2, *args, **kwargs):
    return tree_multimap(lambda x, y: fun(x, y, *args, **kwargs), tree1, tree2)


@jax.jit
def jax_shape_for_update(update, shape_like):
    r"""Reshapes grads from array to tree like structure if neccesary for update

    Args:
        grads: a 1d jax/numpy array
        shape_like: this as in instance having the same type and shape of
                    the desired conversion.

    Returns:
        A possibly non-flat structure of jax arrays containing a copy of data
        compatible with the given shape if jax_available and a copy of update otherwise
    """

    shf, tree = tree_flatten(shape_like)

    updatelist = []
    k = 0
    for s in shf:
        size = s.size
        updatelist.append(jnp.asarray(update[k : k + size]).reshape(s.shape))
        k += size

    return tree_unflatten(tree, updatelist)
