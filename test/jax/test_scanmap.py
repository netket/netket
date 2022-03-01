import pytest

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from functools import partial


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("N", [1, 100])
def test_scan_append_reduce(jit, N):
    def f(x):
        y = jnp.sin(x)
        return y, y, y**2

    x = jnp.linspace(0.0, jnp.pi, 2 * N).reshape((N, 2))

    scan_append_reduce = nk.jax.scan_append_reduce
    if jit:
        scan_append_reduce = jax.jit(scan_append_reduce, static_argnums=(0, 2))

    y, s, s2 = scan_append_reduce(f, x, (True, False, False))
    y_expected = jnp.sin(x)

    np.testing.assert_allclose(y, y_expected)
    np.testing.assert_allclose(s, y_expected.sum(axis=0))
    np.testing.assert_allclose(s2, (y_expected**2).sum(axis=0))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("N", [1, 100])
def test_scanmap(jit, N):

    scan_fun = partial(nk.jax.scan_append_reduce, append_cond=(True, False, False))

    @partial(nk.jax.scanmap, scan_fun=scan_fun, argnums=1)
    def f(c, x):
        y = jnp.sin(x) + c
        return y, y, y**2

    if jit:
        f = jax.jit(f)

    x = jnp.linspace(0.0, jnp.pi, 2 * N).reshape((N, 2))
    c = 1.0

    y, s, s2 = f(c, x)
    y_expected = jnp.sin(x) + c

    np.testing.assert_allclose(y, y_expected)
    np.testing.assert_allclose(s, y_expected.sum(axis=0))
    np.testing.assert_allclose(s2, (y_expected**2).sum(axis=0))
