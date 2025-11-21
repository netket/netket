# Copyright 2025 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for DenseOperator linear operator."""

import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from netket.optimizer.linear_operator import DenseOperator


class TestDenseOperator:
    """Test suite for DenseOperator."""

    def test_construction(self):
        """Test construction with and without diagonal shift."""
        matrix = jnp.eye(5)

        # Without shift
        op = DenseOperator(matrix=matrix)
        assert op.diag_shift == 0.0
        np.testing.assert_array_equal(op.matrix, matrix)

        # With shift
        op_shifted = DenseOperator(matrix=matrix, diag_shift=0.1)
        assert op_shifted.diag_shift == 0.1
        np.testing.assert_array_equal(op_shifted.matrix, matrix)

    def test_matmul(self):
        """Test matrix-vector multiplication with and without shift."""
        matrix = jnp.array([[1, 2], [3, 4]], dtype=float)
        vec = jnp.array([1, 2], dtype=float)

        # Without shift
        op = DenseOperator(matrix=matrix)
        result = op @ vec
        np.testing.assert_allclose(result, matrix @ vec)

        # With shift
        op_shifted = DenseOperator(matrix=matrix, diag_shift=0.5)
        result_shifted = op_shifted @ vec
        np.testing.assert_allclose(result_shifted, matrix @ vec + 0.5 * vec)

    def test_matmul_pytree(self):
        """Test matrix-vector multiplication with PyTree parameters."""
        matrix = jnp.eye(6) * 2.0
        op = DenseOperator(matrix=matrix)

        pytree = {"a": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([4.0, 5.0, 6.0])}
        result = op @ pytree

        vec_flat, unravel = ravel_pytree(pytree)
        expected = unravel(matrix @ vec_flat)

        np.testing.assert_allclose(result["a"], expected["a"])
        np.testing.assert_allclose(result["b"], expected["b"])

    def test_to_dense(self):
        """Test conversion to dense matrix with and without shift."""
        matrix = jnp.array([[1, 2], [3, 4]], dtype=float)

        # Without shift
        op = DenseOperator(matrix=matrix)
        np.testing.assert_array_equal(op.to_dense(), matrix)

        # With shift
        op_shifted = DenseOperator(matrix=matrix, diag_shift=1.5)
        expected = matrix + 1.5 * jnp.eye(2)
        np.testing.assert_allclose(op_shifted.to_dense(), expected)

    def test_operators(self):
        """Test __call__, __add__, and __array__ operators."""
        matrix = jnp.eye(3)
        op = DenseOperator(matrix=matrix, diag_shift=0.1)
        vec = jnp.array([1, 2, 3], dtype=float)

        # __call__ should equal @
        np.testing.assert_allclose(op(vec), op @ vec)

        # __add__ for shift addition
        op_shifted = op + 0.2
        np.testing.assert_allclose(op_shifted.diag_shift, 0.3)

        # __array__ conversion
        arr = op.__array__()
        expected = matrix + 0.1 * jnp.eye(3)
        np.testing.assert_allclose(arr, expected)
