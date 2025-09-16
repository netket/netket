import pytest

import numpy as np
import jax
import jax.numpy as jnp

import netket as nk

from ..common_mesh import with_meshes

@with_meshes(
    auto=[None, ((2,),('A',))]
    )
def test_sort(mesh):
    x = jax.random.randint(jax.random.key(123), (12,), 0, 10)
    print(x.sharding)
    x_sort = nk.jax.sort(x)
    np.testing.assert_array_equal(x_sort, np.sort(x))
    np.testing.assert_array_equal(jax.jit(nk.jax.sort)(x), x_sort)
    assert x.sharding.is_equivalent_to(x_sort.sharding, ndim=1)

    # 2d
    hi = nk.hilbert.Fock(10, 4)
    x = hi.random_state(jax.random.key(123), (12,))
    if not mesh.empty:
        x = jax.device_put(x, jax.sharding.NamedSharding(mesh, jax.P('A')))
    x_sort = nk.jax.sort(x)
    x_i_sort = hi.states_to_numbers(x_sort)
    np.testing.assert_array_equal(x_i_sort, np.sort(x_i_sort))
    # as of sept 2025, it loses track of the sharding.
    # assert x.sharding.is_equivalent_to(x_sort.sharding, ndim=2)


@pytest.mark.parametrize("shape", [(11, 5), (12,)])
def test_searchsorted(shape):
    x = jax.random.randint(jax.random.PRNGKey(123), shape, 0, 10)
    x_sorted = nk.jax.sort(x)

    for i in range(len(x)):
        k = nk.jax.searchsorted(x_sorted, x[i])
        np.testing.assert_array_equal(x_sorted[k], x[i])

    for i, k in enumerate(nk.jax.searchsorted(x_sorted, x)):
        np.testing.assert_array_equal(x_sorted[k], x[i])

    assert nk.jax.searchsorted(x_sorted, x[0]).dtype == np.int32

