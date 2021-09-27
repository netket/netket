import jax
import jax.numpy as jnp
from functools import partial, reduce

from .util import amap, _cumsum

from operator import mul


def _flatten_tensors(x, axes):
    shape = x.shape
    start = _cumsum((0,) + axes[:-1])
    _prod = lambda x: reduce(mul, x, 1)

    def _f(il):
        i, l = il
        return _prod(shape[i : i + l])

    new_shape = tuple(map(_f, zip(start, axes)))
    return x.reshape(new_shape)


# def _flatten_tensors(pt):
#    return amap(_fl, pt.tree, pt.axes)


def to_dense(pt):
    tree_flat = amap(_flatten_tensors, pt.tree, pt.axes)
    for i, td in list(enumerate(pt.treedefs))[::-1]:
        is_leaf = lambda l: jax.tree_structure(l) == td

        def _f(x):
            l, _ = jax.tree_flatten(x)
            return jnp.concatenate(l, axis=i)  # this is the ravel_pytree part

        tree_flat = jax.tree_map(_f, tree_flat, is_leaf=is_leaf)

    return tree_flat
