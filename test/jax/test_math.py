import pytest

from netket.jax import logsumexp_cplx, logdet_cmplx

import jax
import jax.numpy as jnp

import numpy as np

from .. import common

pytestmark = common.skipif_mpi


@pytest.mark.parametrize("a", [[1.0, 2.0], [1.0, -2.0], [1j, 2j]])
@pytest.mark.parametrize("b", [None, [1.0, 2.0], [1.0, -1.0], [1j, -1j]])
def test_logsumexp_cplx(a, b):
    a = jnp.asarray(a)
    if b is not None:
        b = jnp.asarray(b)
        expected = jnp.log(complex(jnp.exp(a[0]) * b[0] + jnp.exp(a[1]) * b[1]))
    else:
        expected = jnp.log(complex(jnp.exp(a[0]) + jnp.exp(a[1])))
    c = logsumexp_cplx(a, b=b)

    assert jnp.iscomplexobj(c)
    np.testing.assert_allclose(c, expected, atol=1e-8)


def test_logdet():
    k = jax.random.PRNGKey(1)

    A = jax.random.normal(k, (3, 4, 5, 5), dtype=jnp.float32)
    ld = logdet_cmplx(A)
    assert ld.shape == (3, 4)
    assert ld.dtype == jnp.complex64
    ldc = logdet_cmplx(A.astype(jnp.complex64))
    assert ldc.dtype == jnp.complex64
    np.testing.assert_allclose(ld, ldc, rtol=1.5e-6)

    A = jax.random.normal(k, (3, 4, 5, 5), dtype=jnp.float64)
    ld = logdet_cmplx(A)
    assert ld.shape == (3, 4)
    assert ld.dtype == jnp.complex128
    ldc = logdet_cmplx(A.astype(jnp.complex128))
    assert ldc.dtype == jnp.complex128
    np.testing.assert_allclose(ld, ldc)
