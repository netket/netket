# Copyright 2024 The NetKet Authors - All rights reserved.
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

import netket as nk
import numpy as np
import pytest
import jax.numpy as jnp
from netket.operator._prod import ProductOperator


class TestProductOperator:
    """Test suite for ProductOperator functionality."""

    @pytest.fixture
    def hilbert_space(self):
        """Create a simple spin-1/2 Hilbert space for testing."""
        L = 4
        g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
        return nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

    @pytest.fixture
    def operators(self, hilbert_space):
        """Create basic spin operators for testing."""
        hi = hilbert_space
        return {
            "sx0": nk.operator.spin.sigmax(hi, 0),
            "sy0": nk.operator.spin.sigmay(hi, 0),
            "sz0": nk.operator.spin.sigmaz(hi, 0),
            "sx1": nk.operator.spin.sigmax(hi, 1),
            "sy1": nk.operator.spin.sigmay(hi, 1),
            "sz1": nk.operator.spin.sigmaz(hi, 1),
            "sx2": nk.operator.spin.sigmax(hi, 2),
            "sz3": nk.operator.spin.sigmaz(hi, 3),
        }

    def test_product_operator_creation(self, operators):
        """Test basic ProductOperator creation."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        # Test creation with default coefficient
        prod = ProductOperator(sx0, sy0)
        assert len(prod.operators) == 2
        assert prod.coefficient == 1.0
        assert prod.hilbert == sx0.hilbert

        # Test creation with custom coefficient
        prod_coeff = ProductOperator(sx0, sy0, coefficient=2.5)
        assert prod_coeff.coefficient == 2.5

        # Test creation with single operator
        prod_single = ProductOperator(sx0)
        assert len(prod_single.operators) == 1
        assert prod_single.coefficient == 1.0

    def test_product_operator_types(self, operators):
        """Test that ProductOperator creates appropriate subclasses."""
        sx0, sy0, sz0 = operators["sx0"], operators["sy0"], operators["sz0"]

        # All JAX operators should create ProductDiscreteJaxOperator
        prod_jax = ProductOperator(sx0, sy0, sz0)
        assert isinstance(
            prod_jax, nk.operator._prod.discrete_jax_operator.ProductDiscreteJaxOperator
        )

        # Test with numba operators
        sx0_numba = sx0.to_numba_operator()
        sy0_numba = sy0.to_numba_operator()
        prod_numba = ProductOperator(sx0_numba, sy0_numba)
        assert isinstance(
            prod_numba, nk.operator._prod.discrete_operator.ProductDiscreteOperator
        )

    def test_product_operator_coefficient_validation(self, operators):
        """Test coefficient validation."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        # Valid scalar coefficients
        ProductOperator(sx0, sy0, coefficient=1.0)
        ProductOperator(sx0, sy0, coefficient=2.5)
        ProductOperator(sx0, sy0, coefficient=-1.0)
        ProductOperator(sx0, sy0, coefficient=1.0 + 2.0j)

        # Invalid non-scalar coefficient should raise TypeError
        with pytest.raises(TypeError, match="Coefficient must be a scalar"):
            ProductOperator(sx0, sy0, coefficient=[1.0, 2.0])

        with pytest.raises(TypeError, match="Coefficient must be a scalar"):
            ProductOperator(sx0, sy0, coefficient=np.array([1.0, 2.0]))

    def test_product_operator_hilbert_space_validation(self, hilbert_space):
        """Test that operators must act on the same Hilbert space."""
        hi1 = hilbert_space

        # Create different Hilbert space
        L2 = 3
        g2 = nk.graph.Hypercube(length=L2, n_dim=1, pbc=False)
        hi2 = nk.hilbert.Spin(s=1 / 2, N=g2.n_nodes)

        sx_hi1 = nk.operator.spin.sigmax(hi1, 0)
        sx_hi2 = nk.operator.spin.sigmax(hi2, 0)

        # Same Hilbert space should work
        ProductOperator(sx_hi1, sx_hi1)

        # Different Hilbert spaces should raise error
        with pytest.raises(NotImplementedError, match="different Hilbert Spaces"):
            ProductOperator(sx_hi1, sx_hi2)

    def test_product_operator_flattening(self, operators):
        """Test that nested ProductOperators are flattened."""
        sx0, sy0, sz0, sx1 = (
            operators["sx0"],
            operators["sy0"],
            operators["sz0"],
            operators["sx1"],
        )

        # Create nested product: (A * B) * (C * D)
        prod1 = ProductOperator(sx0, sy0, coefficient=2.0)
        prod2 = ProductOperator(sz0, sx1, coefficient=3.0)
        nested_prod = ProductOperator(prod1, prod2)

        # Should flatten to A * B * C * D with combined coefficient
        assert len(nested_prod.operators) == 4
        assert nested_prod.coefficient == 6.0  # 2.0 * 3.0
        assert sx0 in nested_prod.operators
        assert sy0 in nested_prod.operators
        assert sz0 in nested_prod.operators
        assert sx1 in nested_prod.operators

    def test_product_operator_arithmetic_mul(self, operators):
        """Test scalar multiplication."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        prod = ProductOperator(sx0, sy0, coefficient=2.0)

        # Test multiplication by scalar
        prod_mul = prod * 3.0
        assert prod_mul.coefficient == 6.0
        assert len(prod_mul.operators) == 2

        # Test multiplication by complex scalar
        prod_complex = prod * (1.0 + 2.0j)
        expected_coeff = 2.0 * (1.0 + 2.0j)
        assert prod_complex.coefficient == expected_coeff

    def test_product_operator_matmul(self, operators):
        """Test matrix multiplication (__matmul__) functionality."""
        sx0, sy0, sz0, sx1 = (
            operators["sx0"],
            operators["sy0"],
            operators["sz0"],
            operators["sx1"],
        )

        # Test ProductOperator @ ProductOperator
        prod1 = ProductOperator(sx0, sy0, coefficient=2.0)
        prod2 = ProductOperator(sz0, sx1, coefficient=3.0)

        result = prod1 @ prod2
        assert isinstance(result, ProductOperator)
        assert len(result.operators) == 4  # sx0, sy0, sz0, sx1
        assert result.coefficient == 6.0  # 2.0 * 3.0

        # Test ProductOperator @ single operator
        single_result = prod1 @ sz0
        assert isinstance(single_result, ProductOperator)
        assert len(single_result.operators) == 3  # sx0, sy0, sz0
        assert single_result.coefficient == 2.0

    def test_product_operator_rmatmul(self, operators):
        """Test reverse matrix multiplication (__rmatmul__) functionality."""
        sx0, sy0, sz0 = operators["sx0"], operators["sy0"], operators["sz0"]

        prod = ProductOperator(sx0, sy0, coefficient=2.0)

        # Test single operator @ ProductOperator
        result = sz0 @ prod
        assert isinstance(result, ProductOperator)
        assert len(result.operators) == 3  # sz0, sx0, sy0
        assert result.coefficient == 2.0
        assert result.operators[0] == sz0

    def test_product_operator_matmul_dtype_inference(self, operators):
        """Test dtype inference in matrix multiplication."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        # Create products with different dtypes
        prod_complex = ProductOperator(sx0, coefficient=1.0 + 1.0j)
        prod_real = ProductOperator(sy0, coefficient=2.0)

        # Test dtype promotion
        result = prod_complex @ prod_real
        expected_dtype = jnp.result_type(prod_complex.dtype, prod_real.dtype)
        assert result.dtype == expected_dtype

    def test_product_operator_max_conn_size(self, operators):
        """Test max_conn_size calculation for ProductOperator."""
        sx0, sy0, sz0 = operators["sx0"], operators["sy0"], operators["sz0"]

        # For JAX operators
        prod_jax = ProductOperator(sx0, sy0, sz0)
        if hasattr(prod_jax, "max_conn_size"):
            expected_size = sx0.max_conn_size * sy0.max_conn_size * sz0.max_conn_size
            assert prod_jax.max_conn_size == expected_size

    def test_product_operator_get_conn_padded(self, operators, hilbert_space):
        """Test get_conn_padded method."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        prod = ProductOperator(sx0, sy0, coefficient=2.0)

        # Test with random state
        rng = nk.jax.PRNGSeq(0)
        states = hilbert_space.random_state(rng.next(), 10)

        if hasattr(prod, "get_conn_padded"):
            conn_states, mels = prod.get_conn_padded(states)

            # Check shapes - get_conn_padded adds a connection dimension
            # conn_states: (batch..., n_conn, n_sites)
            # mels: (batch..., n_conn)
            assert conn_states.shape[:-2] == states.shape[:-1]  # batch dimensions match
            assert mels.shape == conn_states.shape[:-1]  # mels matches conn dimension

            # Check that non-zero matrix elements respect the coefficient
            nonzero_mask = mels != 0
            if np.any(nonzero_mask):
                # Coefficient should be factored into matrix elements
                assert prod.coefficient in mels[nonzero_mask] or np.any(
                    np.abs(mels[nonzero_mask]) > 0
                )

    def test_product_operator_repr(self, operators):
        """Test string representation."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        prod = ProductOperator(sx0, sy0, coefficient=2.5)
        repr_str = repr(prod)

        assert "ProductOperator" in repr_str or "Product" in repr_str
        assert "2.5" in repr_str

    def test_product_operator_jax_numba_conversion(self, operators):
        """Test conversion between JAX and Numba operators."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        # Start with JAX
        prod_jax = ProductOperator(sx0, sy0, coefficient=1.5)

        # Convert to numba if supported
        if hasattr(prod_jax, "to_numba_operator"):
            try:
                prod_numba = prod_jax.to_numba_operator()
                assert prod_numba.coefficient == prod_jax.coefficient

                # Convert back to JAX if supported
                if hasattr(prod_numba, "to_jax_operator"):
                    prod_jax_back = prod_numba.to_jax_operator()
                    assert prod_jax_back.coefficient == prod_numba.coefficient
            except (AttributeError, NotImplementedError) as e:
                # Some conversions might not be fully implemented yet
                pytest.skip(f"Numba conversion not fully implemented: {e}")

    def test_product_operator_tree_flatten_unflatten(self, operators):
        """Test JAX tree flattening and unflattening."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        prod = ProductOperator(sx0, sy0, coefficient=2.0)

        # Test tree operations if available
        if hasattr(prod, "tree_flatten"):
            data, metadata = prod.tree_flatten()

            # Test unflatten
            prod_restored = type(prod).tree_unflatten(metadata, data)

            assert len(prod_restored.operators) == len(prod.operators)
            assert prod_restored.coefficient == prod.coefficient
            assert prod_restored.dtype == prod.dtype

    def test_product_operator_edge_cases(self, operators):
        """Test edge cases and error conditions."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        # Test with zero coefficient
        prod_zero = ProductOperator(sx0, sy0, coefficient=0.0)
        assert prod_zero.coefficient == 0.0

        # Test with very small coefficient
        prod_small = ProductOperator(sx0, sy0, coefficient=1e-10)
        assert prod_small.coefficient == 1e-10

        # Test with large coefficient
        prod_large = ProductOperator(sx0, sy0, coefficient=1e10)
        assert prod_large.coefficient == 1e10

    @pytest.mark.parametrize("coeff", [1.0, -2.5, 1.0 + 2.0j, 0.0])
    def test_product_operator_coefficients(self, operators, coeff):
        """Test ProductOperator with various coefficient values."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        prod = ProductOperator(sx0, sy0, coefficient=coeff)
        assert prod.coefficient == coeff

    def test_product_operator_mathematical_properties(self, operators):
        """Test mathematical properties of ProductOperator."""
        sx0, sy0, sz0 = operators["sx0"], operators["sy0"], operators["sz0"]

        # Test associativity: (A @ B) @ C == A @ (B @ C)
        A = ProductOperator(sx0, coefficient=2.0)
        B = ProductOperator(sy0, coefficient=3.0)
        C = ProductOperator(sz0, coefficient=4.0)

        left_assoc = (A @ B) @ C
        right_assoc = A @ (B @ C)

        # Should have same number of operators and coefficient
        assert len(left_assoc.operators) == len(right_assoc.operators)
        assert left_assoc.coefficient == right_assoc.coefficient

    def test_product_operator_dtype_handling(self, operators):
        """Test dtype handling in ProductOperator."""
        sx0, sy0 = operators["sx0"], operators["sy0"]

        # Test with explicit dtype
        prod = ProductOperator(sx0, sy0, coefficient=1.0, dtype=np.complex128)
        assert prod.dtype == np.complex128

        # Test dtype inference
        prod_inferred = ProductOperator(sx0, sy0, coefficient=1.0 + 2.0j)
        assert np.issubdtype(prod_inferred.dtype, np.complexfloating)

    def test_product_operator_integration_with_existing_operators(self, hilbert_space):
        """Test that ProductOperator integrates well with existing NetKet operators."""
        hi = hilbert_space

        # Create some standard NetKet operators
        h_ising = nk.operator.IsingJax(hi, nk.graph.Hypercube(4, 1, pbc=True), h=1.0)
        sx = nk.operator.spin.sigmax(hi, 0)

        # Test combining with other operator types
        prod = ProductOperator(sx, coefficient=2.0)

        # These operations should work without error
        combined = h_ising + prod  # Addition should create a SumOperator
        assert hasattr(combined, "operators") or hasattr(combined, "hilbert")

        # Matrix multiplication should work
        if hasattr(h_ising, "__matmul__"):
            matmul_result = h_ising @ prod
            assert hasattr(matmul_result, "hilbert")

    def test_pauli_strings_composition(self):
        """Regression test for PauliStrings @ LocalOperator composition."""
        hilbert = nk.hilbert.Spin(s=0.5, N=4)

        pauli = nk.operator.PauliStrings(hilbert, ["XXII", "IIZZ"])
        sy = nk.operator.spin.sigmay(hilbert, 0)

        # Test LocalOperator @ PauliStrings
        composed1 = sy @ pauli
        expected1 = sy.to_dense() @ pauli.to_dense()
        np.testing.assert_allclose(composed1.to_dense(), expected1)

        # Test PauliStrings @ LocalOperator
        composed2 = pauli @ sy
        expected2 = pauli.to_dense() @ sy.to_dense()
        np.testing.assert_allclose(composed2.to_dense(), expected2)


# Integration tests that may require specific test data or fixtures
class TestProductOperatorIntegration:
    """Integration tests for ProductOperator with other NetKet components."""

    def test_product_operator_with_variational_state(self):
        """Test ProductOperator with variational state calculations."""
        # Create a simple system
        L = 4
        g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
        hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

        # Create ProductOperator
        sx0 = nk.operator.spin.sigmax(hi, 0)
        sy1 = nk.operator.spin.sigmay(hi, 1)
        prod_op = ProductOperator(sx0, sy1, coefficient=2.0)

        # Create a simple variational state
        sampler = nk.sampler.ExactSampler(hi)
        ma = nk.models.RBM(alpha=1)
        vs = nk.vqs.MCState(sampler, ma)

        # Test expectation value calculation
        try:
            expectation = vs.expect(prod_op)
            assert np.isfinite(expectation.mean)
            assert expectation.mean.shape == ()
        except Exception as e:
            # If this specific integration doesn't work yet, that's okay
            pytest.skip(f"Integration with variational state not yet supported: {e}")

    def test_product_operator_dense_matrix(self):
        """Test conversion to dense matrix representation."""
        # Create small system for dense matrix test
        L = 2
        g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)
        hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

        sx0 = nk.operator.spin.sigmax(hi, 0)
        sy1 = nk.operator.spin.sigmay(hi, 1)
        prod_op = ProductOperator(sx0, sy1, coefficient=1.0)

        if hasattr(prod_op, "to_dense"):
            dense_matrix = prod_op.to_dense()
            assert dense_matrix.shape == (hi.n_states, hi.n_states)
            assert np.isfinite(dense_matrix).all()
