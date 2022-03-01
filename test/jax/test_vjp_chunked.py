import pytest

import jax
import netket as nk
import numpy as np
from functools import partial


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 16, 10000, 1000000])
@pytest.mark.parametrize("return_forward", [False, True])
@pytest.mark.parametrize("chunk_argnums", [1, (1,)])
@pytest.mark.parametrize("nondiff_argnums", [1, (1,)])
def test_vjp_chunked(chunk_size, jit, return_forward, chunk_argnums, nondiff_argnums):
    @partial(jax.vmap, in_axes=(None, 0))
    def f(p, x):
        return jax.lax.log(p.dot(jax.lax.sin(x)))

    k = jax.random.split(jax.random.PRNGKey(123), 4)
    p = jax.random.uniform(k[0], shape=(8,))
    X = jax.random.uniform(k[2], shape=(10000, 8))
    w = jax.random.uniform(k[3], shape=(10000,))

    vjp_fun_chunked = nk.jax.vjp_chunked(
        f,
        p,
        X,
        chunk_argnums=chunk_argnums,
        chunk_size=chunk_size,
        nondiff_argnums=nondiff_argnums,
        return_forward=return_forward,
    )
    y_expected, vjp_fun = jax.vjp(f, p, X)

    if jit:
        vjp_fun_chunked = jax.jit(vjp_fun_chunked)
        vjp_fun = jax.jit(vjp_fun)

    res_expected = vjp_fun(w)[:1]

    if return_forward:
        y, res = vjp_fun_chunked(w)
        np.testing.assert_allclose(y, y_expected)
    else:
        res = vjp_fun_chunked(w)

    np.testing.assert_allclose(res, res_expected)
