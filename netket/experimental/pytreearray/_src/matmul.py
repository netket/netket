import jax
import jax.numpy as jnp

from .util import _arr_treedef, _treedefs_compose
from . import core


def tree_dot(t1, t2, axes_tree):
    # TODO default withhout axes
    res = jax.tree_util.tree_reduce(jax.lax.add, jax.tree_multimap(jnp.tensordot, t1, t2, axes_tree))
    return res


def _tree_trans(pt, i):
    if pt._is1d():
        return pt
    tdl = _treedefs_compose(*pt.treedefs[:i])
    tdr = _treedefs_compose(*pt.treedefs[i:])
    # TODO flatten & unflatten directly ?
    return jax.tree_transpose(tdl, tdr, pt.tree)


def _matmul(pt1, pt2):
    is_leaf = lambda l: jax.tree_structure(l) == pt1.treedef_r
    tree = jax.tree_map(
        lambda t1: jax.tree_map(
            lambda t2: tree_dot(t1, t2, pt1.axes_r),
            _tree_trans(pt2, 1),
            is_leaf=is_leaf,
        ),
        pt1.tree,
        is_leaf=is_leaf,
    )
    return tree


def matmul(pt1, pt2):
    # contracts the last dim of pt1 with the first dim of pt2
    # TODO eventually replace with a call to tensordot

    assert pt1.treedef_r == pt2.treedef_l
    assert pt1.axes_r == pt2.axes_l  # TODO

    pt1_1d = False
    if pt1._is1d():
        pt1 = pt1.replace(axes=(0,) + pt1.axes, treedefs=(_arr_treedef,) + pt1.treedefs)
        pt1_1d = True

    pt2_1d = False
    if pt2._is1d():
        pt2 = pt2.replace(axes=pt2.axes + (0,), treedefs=pt2.treedefs + (_arr_treedef,))
        pt2_1d = True

    tree = _matmul(pt1, pt2)

    if pt1_1d and pt2_1d:
        return core.PyTreeArray(tree, (), ())
    elif pt1_1d:
        return core.PyTreeArray(tree, pt2.treedefs[1:], pt2.axes[1:])
    elif pt2_1d:
        return core.PyTreeArray(tree, pt1.treedefs[:-1], pt1.axes[:-1])
    else:
        return core.PyTreeArray(tree, pt1.treedefs[:-1] + pt2.treedefs[1:], pt1.axes[:-1] + pt2.axes[1:])
