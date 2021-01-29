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
