import jax.numpy as jnp
import netket as nk
import numpy as np
from functools import partial


def test_scan_append_reduce():
    def f(x):
        y = jnp.sin(x)
        return y, y, y ** 2

    N = 100
    x = jnp.linspace(0.0, jnp.pi, N)

    y, s, s2 = nk.jax.scan_append_reduce(f, x, (True, False, False))
    y_expected = jnp.sin(x)

    np.testing.assert_allclose(y, y_expected)
    np.testing.assert_allclose(s, y_expected.sum())
    np.testing.assert_allclose(s2, (y_expected ** 2).sum())


def test_scanmap():

    scan_fun = partial(nk.jax.scan_append_reduce, append_cond=(True, False, False))

    @partial(nk.jax.scanmap, scan_fun=scan_fun, argnums=1)
    def f(c, x):
        y = jnp.sin(x) + c
        return y, y, y ** 2

    N = 100
    x = jnp.linspace(0.0, jnp.pi, N)
    c = 1.0

    y, s, s2 = f(c, x)
    y_expected = jnp.sin(x) + c

    np.testing.assert_allclose(y, y_expected)
    np.testing.assert_allclose(s, y_expected.sum())
    np.testing.assert_allclose(s2, (y_expected ** 2).sum())
