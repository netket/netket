import pytest

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from functools import partial

from .. import common

pytestmark = common.skipif_distributed


@pytest.mark.parametrize(
    "w_dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.int32]
)
@pytest.mark.parametrize(
    "x_dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.int32]
)
@pytest.mark.parametrize(
    "vec_dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.int32]
)
def test_vjp_chunked_holo_fun(w_dtype, x_dtype, vec_dtype):
    @partial(jax.vmap, in_axes=(None, 0))
    def f(w, x):
        return jnp.sum(w @ x, axis=0)

    w = jnp.ones((10, 20), dtype=w_dtype)
    X = jnp.ones((8, 20), dtype=x_dtype)
    vec = jnp.ones((8,), dtype=vec_dtype)

    res, vjp_fun = nk.jax.vjp(
        f,
        w,
        X,
    )

    dw, dX = vjp_fun(vec)
    np.testing.assert_allclose(w, 1.0)
    np.testing.assert_allclose(X, 1.0)


@pytest.mark.parametrize(
    "w_dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.int32]
)
@pytest.mark.parametrize(
    "x_dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.int32]
)
@pytest.mark.parametrize(
    "vec_dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.int32]
)
def test_vjp_chunked_nonholo_fun(w_dtype, x_dtype, vec_dtype):
    @partial(jax.vmap, in_axes=(None, 0))
    def f(w, x):
        return jnp.sum(w @ x, axis=0) - 1j * jnp.sum(w @ x, axis=0)

    w = jnp.ones((10, 20), dtype=w_dtype)
    X = jnp.ones((8, 20), dtype=x_dtype)
    vec = jnp.ones((8,), dtype=vec_dtype)

    res, vjp_fun = nk.jax.vjp(
        f,
        w,
        X,
    )

    dw, dX = vjp_fun(vec)
    np.testing.assert_allclose(w, 1.0)
    np.testing.assert_allclose(X, 1.0)
