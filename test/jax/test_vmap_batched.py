import pytest

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("batch_size", [None, 32, 10000000])
def test_vmap_batched(batch_size, jit):
    x = jnp.linspace(0.0, 1.0, 100000000).reshape((10000000, 10))

    def f(x):
        assert x.shape == (10,)
        return jnp.sin(x).sum(axis=-1)

    vmap_f = jax.vmap(f)
    vmap_batched_f = nk.jax.vmap_batched(f, batch_size=batch_size)

    if jit:
        vmap_f = jax.jit(vmap_f)
        y_expected = jax.jit(vmap_batched_f)

    y_expected = vmap_f(x)
    y = vmap_batched_f(x)

    np.testing.assert_allclose(y, y_expected)
