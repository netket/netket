import jax
import netket as nk
import numpy as np
from functools import partial


def test_vjp_batched():
    @partial(jax.vmap, in_axes=(None, 0))
    def f(p, x):
        return jax.lax.log(p.dot(jax.lax.sin(x)))

    k = jax.random.split(jax.random.PRNGKey(123), 4)
    p = jax.random.uniform(k[0], shape=(8,))
    # v = jax.random.uniform(k[1], shape=(8,))
    X = jax.random.uniform(k[2], shape=(1024, 8))
    w = jax.random.uniform(k[3], shape=(1024,))

    vjp_fun_batched = nk.jax.vjp_batched(
        f, p, X, batch_argnums=(1,), batch_size=32, nondiff_argnums=1
    )
    vjp_fun = jax.vjp(f, p, X)[1]

    res = vjp_fun_batched(w)
    res_expected = vjp_fun(w)[:1]

    np.testing.assert_allclose(res, res_expected)
