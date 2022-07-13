import pytest

import jax
import jax.numpy as jnp
import netket as nk
import netket.jax as nkjax

import numpy as np
import numpy.testing as npt


def _ones_complex(shape, dtype=jnp.float32):
    xr = jnp.ones(shape, dtype)
    xi = jnp.ones(shape, dtype)
    return xr - 1j * xi


def test_tree_to_real():
    x = _ones_complex((3, 2))
    (xr, xi), reassemble = nkjax.tree_to_real(x)

    assert not jnp.iscomplexobj(xr)
    assert not jnp.iscomplexobj(xi)

    x2 = reassemble((xr, xi))

    assert x2.dtype == x.dtype
    npt.assert_array_equal(x, x2)

    x = {"a": _ones_complex((3, 2)), "b": jnp.ones(2)}
    xr, reassemble = nkjax.tree_to_real(x)

    assert all([not jnp.iscomplexobj(_x) for _x in jax.tree_leaves(xr)])

    x2 = reassemble(xr)

    def _assert_same(x, x2):
        assert x2.dtype == x.dtype
        npt.assert_array_equal(x, x2)

    jax.tree_map(_assert_same, x, x2)
