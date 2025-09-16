import pytest

import numpy as np
import jax
import jax.numpy as jnp

from netket.jax import sort, searchsorted


@pytest.mark.parametrize("shape", [(11, 5), (12,)])
def test_sort(shape):
    x = jax.random.randint(jax.random.PRNGKey(123), shape, 0, 10)
    x_sorted = sort(x)
    for i in range(len(x_sorted) - 1):
        assert x_sorted[i].tolist() <= x_sorted[i + 1].tolist()


@pytest.mark.parametrize("shape", [(11, 5), (12,)])
def test_searchsorted(shape):
    x = jax.random.randint(jax.random.PRNGKey(123), shape, 0, 10)
    x_sorted = sort(x)

    for i in range(len(x)):
        k = searchsorted(x_sorted, x[i])
        np.testing.assert_array_equal(x_sorted[k], x[i])

    for i, k in enumerate(searchsorted(x_sorted, x)):
        np.testing.assert_array_equal(x_sorted[k], x[i])

    assert searchsorted(x_sorted, x[0]).dtype == np.int32


def test_searchsorted_sharding_contexts():
    """Test searchsorted in various contexts (exercises _searchsorted_via_scan internally)."""
    sorted_arr = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    queries = jnp.array([[1, 2, 3], [0, 1, 2]])
    expected = jnp.array([1, 0])

    # Normal usage - this exercises _searchsorted_via_scan with the pvary fix
    result = jax.vmap(lambda q: searchsorted(sorted_arr, q))(queries)
    np.testing.assert_array_equal(result, expected)

    # Test with different input that also exercises the internal function
    result_2d = searchsorted(sorted_arr, queries[0])
    assert result_2d == 1


def test_searchsorted_with_actual_sharding():
    """Test searchsorted with multi-device sharding (reproduces original bug)."""
    if jax.device_count() < 2:
        pytest.skip(f"Test requires at least 2 devices, only {jax.device_count()} available")

    sorted_arr = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    queries = jnp.array([[1, 2, 3], [0, 1, 2], [1, 2, 3], [0, 1, 2]])  # Shape divisible by devices

    @jax.jit
    def vectorized_search(batch_queries):
        return jax.vmap(lambda q: searchsorted(sorted_arr, q))(batch_queries)

    result = vectorized_search(queries)
    expected = jnp.array([1, 0, 1, 0])
    np.testing.assert_array_equal(result, expected)
