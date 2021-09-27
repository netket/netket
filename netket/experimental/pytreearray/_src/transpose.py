import jax
import jax.numpy as jnp

from .util import amap, _cumsum, _flatten, _treedefs_compose
from . import core


def _transpose(x, axes):
    # transpose a tensor with groups of axes whose size is given by axes
    l = _cumsum((0,) + axes[:-1])
    r = _cumsum(axes)
    i = tuple(map(lambda lr: tuple(range(*lr)), zip(l, r)))

    src = _flatten(i)
    dst = _flatten(i[::-1])
    trafo = tuple(map(lambda i: dst.index(i), src))

    return jnp.moveaxis(x, src, trafo)


def transpose(pt):
    tree = amap(_transpose, pt.tree, pt.axes)
    treedefs = pt.treedefs[::-1]

    tree_flat, _ = jax.tree_flatten(tree)
    treedef = _treedefs_compose(*treedefs)
    tree = treedef.unflatten(tree_flat)

    axes = pt.axes[::-1]
    return core.PyTreeArray(tree, treedefs, axes)


# TODO generalize to permute arbitrary axes ([::-1] -> permutation)
