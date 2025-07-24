import pytest

import jax
import netket as nk
import numpy as np
from functools import partial

from netket.jax.sharding import distribute_to_devices_along_axis

from .. import common

pytestmark = common.skipif_distributed


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 16, 123, 8192, 1000000])
@pytest.mark.parametrize("return_forward", [False, True])
@pytest.mark.parametrize("chunk_argnums", [1, (1,)])
@pytest.mark.parametrize("nondiff_argnums", [1, (1,)])
@pytest.mark.parametrize("has_aux", [False, True])
def test_vjp_chunked(
    chunk_size, jit, return_forward, chunk_argnums, nondiff_argnums, has_aux
):
    @partial(jax.vmap, in_axes=(None, 0))
    def f(p, x):
        res = jax.lax.log(p.dot(jax.lax.sin(x)))
        if has_aux:
            aux = {"u": res, "v": jax.lax.cos(res)}
            return res, aux
        return res

    k = jax.random.split(jax.random.PRNGKey(123), 4)
    p = jax.random.uniform(k[0], shape=(8,))
    X = distribute_to_devices_along_axis(jax.random.uniform(k[2], shape=(8192, 8)))
    w = distribute_to_devices_along_axis(jax.random.uniform(k[3], shape=(8192,)))

    vjp_fun_chunked = nk.jax.vjp_chunked(
        f,
        p,
        X,
        chunk_argnums=chunk_argnums,
        chunk_size=chunk_size,
        nondiff_argnums=nondiff_argnums,
        return_forward=return_forward,
        has_aux=has_aux,
    )
    y_expected, vjp_fun, *aux_expected = jax.vjp(f, p, X, has_aux=has_aux)

    if jit:
        vjp_fun_chunked = jax.jit(vjp_fun_chunked)
        vjp_fun = jax.jit(vjp_fun)

    res_expected = vjp_fun(w)[:1]

    if return_forward:
        y, res, *aux = vjp_fun_chunked(w)
        np.testing.assert_allclose(y, y_expected)
    else:
        res = vjp_fun_chunked(w)
        if has_aux:
            res, *aux = res
        else:
            aux = []
    jax.tree_util.tree_map(np.testing.assert_allclose, aux, aux_expected)
    np.testing.assert_allclose(res, res_expected)
