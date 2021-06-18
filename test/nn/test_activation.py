import pytest

import jax
import jax.numpy as jnp
import numpy as np

import netket as nk

from .. import common

SEED = 123
pytestmark = common.skipif_mpi


@pytest.mark.parametrize("rdtype", [jnp.float64])
def test_log_cosh(rdtype):
    N = 100
    x = jnp.sort(jax.random.uniform(jax.random.PRNGKey(SEED), (N,), dtype=rdtype) * 5)

    # this rtol should be increased
    np.testing.assert_allclose(
        nk.nn.log_cosh(x + 0j), nk.nn.log_cosh(x) + 0j, rtol=5e-6
    )

    # netket#768
    yi = nk.nn.log_cosh(0 + 1j * x)
    yr = jnp.log(jnp.cos(x) + 0.0j)
    np.testing.assert_allclose(yi.real, yr.real)
