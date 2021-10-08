import jax
import jax.numpy as jnp
import netket as nk
import numpy as np


def test_vmap_batched():
    x = jnp.linspace(0.0, 1.0, 1000).reshape((100, 10))

    def f(x):
        assert x.shape == (10,)
        return jnp.sin(x).sum(axis=-1)

    y = jax.vmap(f)(x)
    y_expected = nk.jax.vmap_batched(f, batch_size=5)(x)
    np.testing.assert_allclose(y, y_expected)
