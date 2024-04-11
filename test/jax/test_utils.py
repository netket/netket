import pytest

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import operator


@pytest.mark.parametrize(
    "tree",
    [
        {"a": 1, "b": (1, 1 + 2.0j, 3.0j)},
        {
            "a": jnp.ones(100),
            "b": (
                jnp.ones((1, 2)) + 3.0j,
                jnp.ones(3),
                (jnp.ones((3, 1)) + 3.0j, jnp.ones((2, 2)) + 3.0j),
            ),
        },
    ],
)
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
