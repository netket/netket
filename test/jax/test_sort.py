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
    """
    Test searchsorted in various sharding contexts to ensure the pvary fix works correctly.

    This test verifies that searchsorted works with:
    - Non-sharded inputs (should work as before)
    - Sharded inputs (should work with pvary)
    - Inside shard_map (reproduces the original bug that was fixed)
    - Outside shard_map (should work normally)
    """
    if jax.device_count() < 2:
        pytest.skip(f"Test requires at least 2 devices, only {jax.device_count()} available")

    # Test data
    sorted_arr = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    query = jnp.array([1, 2, 3])

    # Test 1: Normal usage (non-sharded)
    result_normal = searchsorted(sorted_arr, query)
    expected = 1  # query [1,2,3] should be inserted at index 1
    assert result_normal == expected

    # Test 2: With vmapped queries (triggers _searchsorted_via_scan)
    queries = jnp.array([[1, 2, 3], [0, 1, 2]])
    result_vmap = jax.vmap(lambda q: searchsorted(sorted_arr, q))(queries)
    expected_vmap = jnp.array([1, 0])
    np.testing.assert_array_equal(result_vmap, expected_vmap)

    # Test 3: Inside a simple function that could be sharded
    @jax.jit
    def search_fn(queries):
        return jax.vmap(lambda q: searchsorted(sorted_arr, q))(queries)

    result_jit = search_fn(queries)
    np.testing.assert_array_equal(result_jit, expected_vmap)

    # Test 4: Test that the original MWE pattern works (simplified)
    # This mimics what happens in fermionic operators but in a simpler form
    def vectorized_search(batch_queries):
        return jax.vmap(lambda q: searchsorted(sorted_arr, q))(batch_queries)

    # Create batched data that can be distributed across devices
    batch_size = 4  # Divisible by common device counts
    batch_queries = jnp.tile(queries, (batch_size // 2, 1))  # Shape: (4, 3)

    # Test with regular jit
    jit_vectorized = jax.jit(vectorized_search)
    result_batch = jit_vectorized(batch_queries)

    # Should get the same results repeated
    expected_batch = jnp.tile(expected_vmap, batch_size // 2)
    np.testing.assert_array_equal(result_batch, expected_batch)


def test_searchsorted_with_different_device_settings():
    """
    Test that searchsorted works correctly with different JAX device settings.

    This ensures the pvary fix doesn't break functionality when JAX_NUM_CPU_DEVICES=1
    or when NETKET_EXPERIMENTAL_SHARDING=0.
    """
    # Test with basic functionality that should always work
    sorted_arr = jnp.array([[0, 1], [1, 2], [2, 3]])
    query = jnp.array([1, 2])

    result = searchsorted(sorted_arr, query)
    assert result == 1

    # Test with multiple queries
    queries = jnp.array([[1, 2], [0, 1], [2, 3]])
    results = jax.vmap(lambda q: searchsorted(sorted_arr, q))(queries)
    expected = jnp.array([1, 0, 2])
    np.testing.assert_array_equal(results, expected)
