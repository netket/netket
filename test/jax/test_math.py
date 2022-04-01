import pytest

from netket.jax import logsumexp_cplx
import jax.numpy as jnp
from numpy.testing import assert_allclose


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
    assert_allclose(c, expected, atol=1e-8)
