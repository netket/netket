import pytest

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from .. import common

pytestmark = common.skipif_distributed


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 16, 123, 10000, 1000000])
def test_vmap_chunked(chunk_size, jit):
    x = jnp.linspace(0.0, 1.0, 100000).reshape((10000, 10))

    def f(x):
        assert x.shape == (10,)
        return jnp.sin(x).sum(axis=-1)

    vmap_f = jax.vmap(f)
    vmap_chunked_f = nk.jax.vmap_chunked(f, chunk_size=chunk_size)

    if jit:
        vmap_f = jax.jit(vmap_f)
        y_expected = jax.jit(vmap_chunked_f)

    y_expected = vmap_f(x)
    y = vmap_chunked_f(x)

    np.testing.assert_allclose(y, y_expected)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 16, 123, 10000, 1000000])
def test_apply_chunked(chunk_size, jit):
    x = jnp.linspace(0.0, 1.0, 100000).reshape((10000, 10))

    def f(x):
        assert x.shape[-1] == 10
        return jnp.sin(x).sum(axis=-1)

    chunked_f = nk.jax.apply_chunked(f, in_axes=(0,), chunk_size=chunk_size)

    if jit:
        f = jax.jit(f)
        y_expected = jax.jit(chunked_f)

    y_expected = f(x)
    y = chunked_f(x)

    np.testing.assert_allclose(y, y_expected)
