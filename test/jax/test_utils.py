import pytest

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import operator

trees = [
    {"a": 1, "b": (1, 1 + 2.0j, 3.0j)},
    {
        "a": jnp.ones(100),
        "b": (
            jnp.ones((1, 2)) + 3.0j,
            jnp.ones(3),
            (jnp.ones((3, 1)) + 3.0j, jnp.ones((2, 2)) + 3.0j),
        ),
    },
]


@pytest.mark.parametrize("tree", trees)
# @pytest.mark.parametrize("jit", [False, True])
def test_tree_to_real(tree):
    tree_real, restore = nk.jax.tree_to_real(tree)
    tree_restored = restore(tree_real)

    assert not jax.tree_util.tree_reduce(
        operator.or_, jax.tree_util.tree_map(jnp.iscomplexobj, tree_real)
    )
    assert jax.tree_util.tree_structure(tree) == jax.tree_util.tree_structure(
        tree_restored
    )
    jax.tree_util.tree_map(np.testing.assert_allclose, tree_restored, tree)


@pytest.mark.parametrize("tree", trees)
def test_tree_norm(tree):
    tree_norm = nk.jax.tree_norm(tree)
    ravel_norm = jnp.linalg.norm(nk.jax.tree_ravel(tree)[0])
    np.testing.assert_allclose(tree_norm, ravel_norm)

    for p in [0.5, 1, 2, 3]:
        tree_norm = nk.jax.tree_norm(tree, ord=p)
        ravel_norm = jnp.linalg.norm(nk.jax.tree_ravel(tree)[0], ord=p)
        np.testing.assert_allclose(tree_norm, ravel_norm)
