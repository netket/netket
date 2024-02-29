import pytest

import numpy as np
import jax

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
