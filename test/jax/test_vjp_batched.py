import pytest

import jax
import netket as nk
import numpy as np
from functools import partial


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("batch_size", [None, 32, 10000000])
@pytest.mark.parametrize("return_forward", [False, True])
def test_vjp_batched(batch_size, jit, return_forward):
    @partial(jax.vmap, in_axes=(None, 0))
    def f(p, x):
        return jax.lax.log(p.dot(jax.lax.sin(x)))

    k = jax.random.split(jax.random.PRNGKey(123), 4)
    p = jax.random.uniform(k[0], shape=(8,))
    X = jax.random.uniform(k[2], shape=(10000000, 8))
    w = jax.random.uniform(k[3], shape=(10000000,))

    vjp_fun_batched = nk.jax.vjp_batched(
        f,
        p,
        X,
        batch_argnums=(1,),
        batch_size=batch_size,
        nondiff_argnums=1,
        return_forward=return_forward,
    )
    y_expected, vjp_fun = jax.vjp(f, p, X)

    if jit:
        vjp_fun_batched = jax.jit(vjp_fun_batched)
        vjp_fun = jax.jit(vjp_fun)

    res_expected = vjp_fun(w)[:1]

    if return_forward:
        y, res = vjp_fun_batched(w)
        np.testing.assert_allclose(y, y_expected)
    else:
        res = vjp_fun_batched(w)

    np.testing.assert_allclose(res, res_expected)
