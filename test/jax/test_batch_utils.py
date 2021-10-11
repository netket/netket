import pytest

import jax.numpy as jnp
import netket as nk


@pytest.mark.parametrize("batch_size", [2, None])
def test_batch(batch_size):
    x = jnp.ones((10, 2, 3))
    x_batched, unbatch_fn = nk.jax.batch(x, batch_size=batch_size)
    if batch_size is None:
        assert x_batched.shape == (1, 10, 2, 3)
    else:
        assert x_batched.shape == (5, batch_size, 2, 3)
    assert x.shape == unbatch_fn(x_batched).shape


@pytest.mark.parametrize("batch_size", [2, None])
def test_batch_pytree(batch_size):
    x = {"a": jnp.ones((10, 2, 3)), "b": jnp.ones((10, 4, 1))}
    x_batched, unbatch_fn = nk.jax.batch(x, batch_size=batch_size)
    if batch_size is None:
        assert x_batched["a"].shape == (1, 10, 2, 3)
        assert x_batched["b"].shape == (1, 10, 4, 1)
    else:
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
