import pytest

import jax.numpy as jnp
import netket as nk


@pytest.mark.parametrize("chunk_size", [2, None])
def test_chunk(chunk_size):
    x = jnp.ones((10, 2, 3))
    x_chunked, unchunk_fn = nk.jax.chunk(x, chunk_size=chunk_size)
    if chunk_size is None:
        assert x_chunked.shape == (1, 10, 2, 3)
    else:
        assert x_chunked.shape == (5, chunk_size, 2, 3)
    assert x.shape == unchunk_fn(x_chunked).shape


@pytest.mark.parametrize("chunk_size", [2, None])
def test_chunk_pytree(chunk_size):
    x = {"a": jnp.ones((10, 2, 3)), "b": jnp.ones((10, 4, 1))}
    x_chunked, unchunk_fn = nk.jax.chunk(x, chunk_size=chunk_size)
    if chunk_size is None:
        assert x_chunked["a"].shape == (1, 10, 2, 3)
        assert x_chunked["b"].shape == (1, 10, 4, 1)
    else:
        assert x_chunked["a"].shape == (5, 2, 2, 3)
        assert x_chunked["b"].shape == (5, 2, 4, 1)
    assert x["a"].shape == unchunk_fn(x_chunked)["a"].shape
    assert x["b"].shape == unchunk_fn(x_chunked)["b"].shape


def test_unchunk():
    x_chunked = jnp.ones((10, 2, 3))
    x, chunk_fn = nk.jax.unchunk(x_chunked)
    assert x.shape == (20, 3)
    assert x_chunked.shape == chunk_fn(x).shape


def test_unchunk_pytree():
    x_chunked = {"a": jnp.ones((10, 2, 3)), "b": jnp.ones((10, 2, 1))}
    x, chunk_fn = nk.jax.unchunk(x_chunked)
    assert x["a"].shape == (20, 3)
    assert x["b"].shape == (20, 1)
    assert x_chunked["a"].shape == chunk_fn(x)["a"].shape
    assert x_chunked["b"].shape == chunk_fn(x)["b"].shape
