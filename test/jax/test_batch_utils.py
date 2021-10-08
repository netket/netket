import jax.numpy as jnp
import netket as nk


def test_batch():
    x = jnp.ones((10, 2, 3))
    x_batched, unbatch_fn = nk.jax.batch(x, batch_size=2)
    assert x_batched.shape == (5, 2, 2, 3)
    assert x.shape == unbatch_fn(x_batched).shape


def test_batch_pytree():
    x = {"a": jnp.ones((10, 2, 3)), "b": jnp.ones((10, 4, 1))}
    x_batched, unbatch_fn = nk.jax.batch(x, batch_size=2)
    assert x_batched["a"].shape == (5, 2, 2, 3)
    assert x_batched["b"].shape == (5, 2, 4, 1)
    assert x["a"].shape == unbatch_fn(x_batched)["a"].shape
    assert x["b"].shape == unbatch_fn(x_batched)["b"].shape


def test_unbatch():
    x_batched = jnp.ones((10, 2, 3))
    x, batch_fn = nk.jax.unbatch(x_batched)
    assert x.shape == (20, 3)
    assert x_batched.shape == batch_fn(x).shape


def test_unbatch_pytree():
    x_batched = {"a": jnp.ones((10, 2, 3)), "b": jnp.ones((10, 2, 1))}
    x, batch_fn = nk.jax.unbatch(x_batched)
    assert x["a"].shape == (20, 3)
    assert x["b"].shape == (20, 1)
    assert x_batched["a"].shape == batch_fn(x)["a"].shape
    assert x_batched["b"].shape == batch_fn(x)["b"].shape
