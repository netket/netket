# Copyright 2018 The JAX Authors.
# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from jax/jax/_src/numpy/lax_numpy.py

# Here we add a lexicographic searchsorted, i.e. one which supports vectors as keys.
# This is currently not implemented in jax (nor in numpy).
# Our main use case is to find of rows in a matrix in log time.

import jax
import numpy as np
from functools import partial
import jax.numpy as jnp

from netket.utils.types import Array


def _sort_lexicographic(x):
    assert x.ndim == 2
    perm = jnp.lexsort(list(x.T)[::-1])
    return x[perm]


@jax.jit
def sort(x: Array) -> Array:
    """Lexicographically sort the rows of a matrix, taking the columns as sequences of keys

    Args:
        x: 1D/2D Input array
    Returns:
        A sorted copy of x

    Example:
        >>> import jax.numpy as jnp
        >>> from netket.jax import sort
        >>> x = jnp.array([[1,2,3], [0,2,2], [0,1,2]])
        >>> sort(x)
        Array([[0, 1, 2],
               [0, 2, 2],
               [1, 2, 3]], dtype=int64)
    """
    if x.ndim == 1:
        return jnp.sort(x)
    else:
        return _sort_lexicographic(x)


def _less_equal_lexicographic(x_keys, y_keys):
    assert x_keys.shape == y_keys.shape
    assert x_keys.dtype == y_keys.dtype
    p = None
    for xk, yk in zip(x_keys[::-1], y_keys[::-1]):
        p = (
            jax.lax.bitwise_or(
                jax.lax.lt(xk, yk), jax.lax.bitwise_and(jax.lax.eq(xk, yk), p)
            )
            if p is not None
            else jax.lax.le(xk, yk)
        )
    return p


@partial(jnp.vectorize, signature="(n)->()", excluded={0, 2, 3})
def _searchsorted_via_scan(sorted_arr, query, dtype, op):
    def body_fun(_, state):
        low, high = state
        mid = jax.lax.div(low + high, jnp.full_like(low, 2))
        go_left = op(query, sorted_arr[mid])
        return jax.lax.select(go_left, low, mid), jax.lax.select(go_left, mid, high)

    n = len(sorted_arr)
    n_levels = int(np.ceil(np.log2(n + 1)))
    shape = query.shape[:-1]
    init = jnp.full(shape, dtype(0)), jnp.full(shape, dtype(n))
    return jax.lax.fori_loop(0, n_levels, body_fun, init)[1]


def _searchsorted_lexicographic(a, v):
    assert a.ndim == 2
    assert v.ndim >= 1
    assert a.shape[-1] == v.shape[-1]
    dtype = np.int32 if len(a) <= np.iinfo(np.int32).max else np.int64
    a = a.astype(jnp.promote_types(a, v))
    v = v.astype(jnp.promote_types(a, v))
    return _searchsorted_via_scan(a, v, dtype, _less_equal_lexicographic)


@partial(jax.jit)
def searchsorted(a: Array, v: Array) -> Array:
    """Find the indices where rows should be inserted into a matrix to maintain lexicographic order.

    Args:
        a: 1D/2D Input array. Rows must me sorted lexicographically in ascending order.
        v: Array of rows to insert into a
    Returns:
        A integer array of row indices with shape v.shape[:-1]

    Example:
        >>> import jax.numpy as jnp
        >>> from netket.jax import searchsorted
        >>> a = jnp.array([[0,1,2], [0,2,2], [1,2,3]])
        >>> v = jnp.array([[0,2,2]])
        >>> searchsorted(a, v)
        Array([1], dtype=int32)
    """
    if a.ndim == 1:
        return jnp.searchsorted(a, v)
    else:
        return _searchsorted_lexicographic(a, v)
