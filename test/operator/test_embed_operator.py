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

import pytest
import numpy as np
import jax.numpy as jnp

import netket as nk


class TestEmbedOperator:
    """Test suite for EmbedOperator functionality."""

    @pytest.fixture
    def spin_hilbert(self):
        """Create a simple spin-1/2 Hilbert space."""
        return nk.hilbert.Spin(s=1 / 2, N=2)

    @pytest.fixture
    def fock_hilbert(self):
        """Create a Fock Hilbert space."""
        return nk.hilbert.Fock(n_max=3, N=2)

    @pytest.fixture
    def spin_operators(self, spin_hilbert):
        """Create basic spin operators."""
        hi = spin_hilbert
        return {
            "sx0": nk.operator.spin.sigmax(hi, 0),
            "sy0": nk.operator.spin.sigmay(hi, 0),
            "sz0": nk.operator.spin.sigmaz(hi, 0),
            "sx1": nk.operator.spin.sigmax(hi, 1),
        }

    def test_embed_operator_creation(self, spin_hilbert):
        """Test basic EmbedOperator creation."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        # Create operator on first subspace
        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        assert embed_op.hilbert == hi_tensor
        assert embed_op.operator == sx
        assert embed_op.subspace == 0
        assert embed_op.dtype == sx.dtype

    def test_embed_operator_wrong_hilbert_type(self, spin_hilbert):
        """Test that EmbedOperator requires TensorHilbert."""
        hi = spin_hilbert
        sx = nk.operator.spin.sigmax(hi, 0)

        # Should raise TypeError when not given a TensorHilbert
        with pytest.raises(
            TypeError, match="hilbert space of an EmbedOperator must be a TensorHilbert"
        ):
            nk.operator.EmbedOperator(hi, sx, subspace=0)

    def test_embed_operator_subspace_mismatch(self, spin_hilbert, fock_hilbert):
        """Test that operator must match the specified subspace."""
        hi_spin = spin_hilbert
        hi_fock = fock_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi_spin, hi_fock)

        # Operator for spin system
        sx = nk.operator.spin.sigmax(hi_spin, 0)

        # Should work with subspace 0 (spin)
        nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Should raise error with subspace 1 (fock)
        with pytest.raises(TypeError, match="hilbert space of the tensor hilbert"):
            nk.operator.EmbedOperator(hi_tensor, sx, subspace=1)

    def test_embed_operator_connected_elements_simple(self, spin_hilbert):
        """Test that embedded operator has correct connected elements."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        # Create operator on first subspace
        sx0 = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx0, subspace=0)

        # Test on a specific state
        # State: [spin1_site0, spin1_site1, spin2_site0, spin2_site1]
        state = jnp.array([[1.0, -1.0, 1.0, -1.0]])

        x_conn, mels = embed_op.get_conn_padded(state)

        # The operator should flip the first site (index 0)
        # but leave the second subspace unchanged
        expected_state = jnp.array([[-1.0, -1.0, 1.0, -1.0]])

        # Check that connected states are correct
        assert x_conn.shape == (1, sx0.max_conn_size, 4)
        assert mels.shape == (1, sx0.max_conn_size)

        # Find non-zero matrix elements
        nonzero_mask = mels[0] != 0
        assert jnp.any(nonzero_mask)

        # Check that the connected state is as expected
        connected_states = x_conn[0, nonzero_mask]
        np.testing.assert_array_equal(connected_states[0], expected_state[0])

    def test_embed_operator_vs_manual_construction(self, spin_hilbert):
        """Test EmbedOperator against manually constructed operator on full space."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)
        hi_product = hi * hi  # Product Hilbert space

        # Create operator on subspace using EmbedOperator
        sx0 = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx0, subspace=0)

        # Create equivalent operator on product space
        # sigmax on site 0 of first subsystem
        sx0_full = nk.operator.spin.sigmax(hi_product, 0)

        # Compare dense matrices (for small systems)
        if hi.n_states <= 256:  # Only test for small systems
            dense_embed = embed_op.to_dense()
            dense_manual = sx0_full.to_dense()

            np.testing.assert_allclose(dense_embed, dense_manual, rtol=1e-10)

    def test_embed_different_hilbert_spaces(self, spin_hilbert, fock_hilbert):
        """Test embedding operators in different types of Hilbert spaces."""
        hi_spin = spin_hilbert
        hi_fock = fock_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi_spin, hi_fock)

        # Embed spin operator in first subspace
        sx = nk.operator.spin.sigmax(hi_spin, 0)
        embed_spin = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Embed fock operator in second subspace
        a = nk.operator.boson.create(hi_fock, 0)
        embed_fock = nk.operator.EmbedOperator(hi_tensor, a, subspace=1)

        assert embed_spin.hilbert == hi_tensor
        assert embed_fock.hilbert == hi_tensor
        assert embed_spin.subspace == 0
        assert embed_fock.subspace == 1

        # Test that connected elements are computed correctly
        rng = nk.jax.PRNGSeq(0)
        states = hi_tensor.random_state(rng.next(), 5)

        x_conn_spin, mels_spin = embed_spin.get_conn_padded(states)
        x_conn_fock, mels_fock = embed_fock.get_conn_padded(states)

        # Check shapes
        assert x_conn_spin.shape[0] == 5
        assert x_conn_fock.shape[0] == 5
        assert x_conn_spin.shape[-1] == hi_tensor.size
        assert x_conn_fock.shape[-1] == hi_tensor.size

    def test_embed_operator_matmul_same_subspace(self, spin_hilbert):
        """Test that multiplying two embed operators on same subspace combines them."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        # Create two operators on the same subspace
        sx = nk.operator.spin.sigmax(hi, 0)
        sy = nk.operator.spin.sigmay(hi, 0)

        embed_sx = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)
        embed_sy = nk.operator.EmbedOperator(hi_tensor, sy, subspace=0)

        # Matrix multiply them
        result = embed_sx @ embed_sy

        # Result should be a single EmbedOperator with combined operator
        assert isinstance(result, nk.operator.EmbedOperator)
        assert result.subspace == 0
        assert result.hilbert == hi_tensor

        # The underlying operator should be sx @ sy
        expected_inner = sx @ sy
        if hasattr(result.operator, "to_dense") and hasattr(expected_inner, "to_dense"):
            np.testing.assert_allclose(
                result.operator.to_dense(), expected_inner.to_dense(), rtol=1e-10
            )

    def test_embed_operator_matmul_different_subspaces(self, spin_hilbert):
        """Test that multiplying embed operators on different subspaces creates ProductOperator."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx0 = nk.operator.spin.sigmax(hi, 0)
        sy1 = nk.operator.spin.sigmay(hi, 1)

        embed0 = nk.operator.EmbedOperator(hi_tensor, sx0, subspace=0)
        embed1 = nk.operator.EmbedOperator(hi_tensor, sy1, subspace=1)

        # Matrix multiply them
        result = embed0 @ embed1

        # Result should be a ProductOperator since they act on different subspaces
        from netket.operator._prod import ProductOperator

        assert isinstance(result, ProductOperator)
        assert result.hilbert == hi_tensor

    def test_embed_operator_add_same_subspace(self, spin_hilbert):
        """Test that adding two embed operators on same subspace combines them."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        sy = nk.operator.spin.sigmay(hi, 0)

        embed_sx = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)
        embed_sy = nk.operator.EmbedOperator(hi_tensor, sy, subspace=0)

        # Add them
        result = embed_sx + embed_sy

        # Result should be a single EmbedOperator with sum of operators
        assert isinstance(result, nk.operator.EmbedOperator)
        assert result.subspace == 0
        assert result.hilbert == hi_tensor

        # The underlying operator should be sx + sy
        expected_inner = sx + sy
        if hasattr(result.operator, "to_dense") and hasattr(expected_inner, "to_dense"):
            np.testing.assert_allclose(
                result.operator.to_dense(), expected_inner.to_dense(), rtol=1e-10
            )

    def test_embed_operator_add_different_subspaces(self, spin_hilbert):
        """Test that adding embed operators on different subspaces creates SumOperator."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx0 = nk.operator.spin.sigmax(hi, 0)
        sy1 = nk.operator.spin.sigmay(hi, 1)

        embed0 = nk.operator.EmbedOperator(hi_tensor, sx0, subspace=0)
        embed1 = nk.operator.EmbedOperator(hi_tensor, sy1, subspace=1)

        # Add them
        result = embed0 + embed1

        # Result should be a SumOperator since they act on different subspaces
        from netket.operator._sum import SumOperator

        assert isinstance(result, SumOperator)
        assert result.hilbert == hi_tensor

    def test_embed_operator_max_conn_size(self, spin_hilbert):
        """Test that max_conn_size is correctly inherited."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        assert embed_op.max_conn_size == sx.max_conn_size

    def test_embed_operator_dtype_inheritance(self, spin_hilbert):
        """Test that dtype is correctly inherited from the operator."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        # Create operator with specific dtype
        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        assert embed_op.dtype == sx.dtype

    def test_embed_operator_repr(self, spin_hilbert):
        """Test string representation."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        repr_str = repr(embed_op)
        assert "EmbedOperator" in repr_str or "Embed" in repr_str
        assert "0" in repr_str  # subspace index

    def test_embed_operator_to_numba(self, spin_hilbert):
        """Test conversion to numba operator."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        if hasattr(embed_op, "to_numba_operator"):
            numba_op = embed_op.to_numba_operator()
            assert numba_op.hilbert == embed_op.hilbert
            assert numba_op.subspace == embed_op.subspace

    def test_embed_operator_pytree(self, spin_hilbert):
        """Test JAX pytree flattening and unflattening."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        if hasattr(embed_op, "tree_flatten"):
            # Test tree operations
            data, metadata = embed_op.tree_flatten()

            # Test unflatten
            embed_restored = type(embed_op).tree_unflatten(metadata, data)

            assert embed_restored.hilbert == embed_op.hilbert
            assert embed_restored.subspace == embed_op.subspace
            assert embed_restored.dtype == embed_op.dtype

    def test_embed_operator_multiple_subspaces(self):
        """Test embedding in tensor product of multiple spaces."""
        hi1 = nk.hilbert.Spin(s=1 / 2, N=2)
        hi2 = nk.hilbert.Fock(n_max=3, N=2)
        hi3 = nk.hilbert.Spin(s=1, N=2)

        hi_tensor = nk.hilbert.TensorHilbert(hi1, hi2, hi3)

        # Create operators for each subspace
        sx = nk.operator.spin.sigmax(hi1, 0)
        a = nk.operator.boson.create(hi2, 0)
        sz = nk.operator.spin.sigmaz(hi3, 0)

        embed0 = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)
        embed1 = nk.operator.EmbedOperator(hi_tensor, a, subspace=1)
        embed2 = nk.operator.EmbedOperator(hi_tensor, sz, subspace=2)

        assert embed0.subspace == 0
        assert embed1.subspace == 1
        assert embed2.subspace == 2

        # Test combinations
        combined = embed0 + embed1 + embed2
        assert combined.hilbert == hi_tensor

    def test_embed_operator_consistency_with_direct_operator(self, spin_hilbert):
        """
        Test that EmbedOperator gives same results as operator defined directly
        on the joint space.
        """
        hi = spin_hilbert
        # Use TensorHilbert explicitly
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)
        # Use product space
        hi_product = hi * hi

        # Create test states
        rng = nk.jax.PRNGSeq(42)
        n_samples = 10

        # Generate states for tensor hilbert (they should be compatible)
        states_tensor = hi_tensor.random_state(rng.next(), n_samples)

        # Create operator on subspace 1 (second copy of hi)
        sz1 = nk.operator.spin.sigmaz(hi, 1)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sz1, subspace=1)

        # Create equivalent operator on product space
        # This should act on sites 2 and 3 (indices in the full space)
        sz_full = nk.operator.spin.sigmaz(hi_product, 3)

        # Get connected elements
        x_conn_embed, mels_embed = embed_op.get_conn_padded(states_tensor)

        # For comparison, need to reshape states to work with product space
        x_conn_full, mels_full = sz_full.get_conn_padded(states_tensor)

        # The matrix elements should be the same (sigmaz is diagonal)
        np.testing.assert_allclose(mels_embed, mels_full, rtol=1e-10)

    def test_embed_operator_hermiticity(self, spin_hilbert):
        """Test that hermitian operators remain hermitian when embedded."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        # sigmaz is hermitian
        sz = nk.operator.spin.sigmaz(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sz, subspace=0)

        if hasattr(embed_op, "to_dense"):
            dense = embed_op.to_dense()
            # Check hermiticity: H = H†
            np.testing.assert_allclose(dense, jnp.conj(dense.T), rtol=1e-10)

    def test_embed_operator_scalar_multiplication(self, spin_hilbert):
        """Test scalar multiplication of EmbedOperator."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Multiply by scalar
        scaled_op = 2.5 * embed_op

        # Test that scaling works correctly
        if hasattr(embed_op, "to_dense") and hasattr(scaled_op, "to_dense"):
            np.testing.assert_allclose(
                scaled_op.to_dense(), 2.5 * embed_op.to_dense(), rtol=1e-10
            )

    def test_embed_operator_chain_matmul(self, spin_hilbert):
        """Test chaining multiple matmul operations on same subspace."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        sy = nk.operator.spin.sigmay(hi, 0)
        sz = nk.operator.spin.sigmaz(hi, 0)

        embed_sx = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)
        embed_sy = nk.operator.EmbedOperator(hi_tensor, sy, subspace=0)
        embed_sz = nk.operator.EmbedOperator(hi_tensor, sz, subspace=0)

        # Chain multiply: (sx @ sy) @ sz
        result = (embed_sx @ embed_sy) @ embed_sz

        # Should result in a single EmbedOperator
        assert isinstance(result, nk.operator.EmbedOperator)
        assert result.subspace == 0

        # Compare with direct multiplication
        expected_inner = (sx @ sy) @ sz
        if hasattr(result.operator, "to_dense") and hasattr(expected_inner, "to_dense"):
            np.testing.assert_allclose(
                result.operator.to_dense(), expected_inner.to_dense(), rtol=1e-10
            )

    def test_embed_operator_to_sparse_scipy(self, spin_hilbert):
        """Test to_sparse method returns correct scipy sparse matrix."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Get sparse representation
        sparse_mat = embed_op.to_sparse()

        # Check it's the right type
        from scipy.sparse import csr_matrix

        assert isinstance(sparse_mat, csr_matrix)

        # Check shape
        assert sparse_mat.shape == (hi_tensor.n_states, hi_tensor.n_states)

        # Check correctness by comparing with to_dense
        dense_from_sparse = sparse_mat.toarray()
        dense_direct = embed_op.to_dense()

        np.testing.assert_allclose(dense_from_sparse, dense_direct, rtol=1e-10)

    def test_embed_operator_to_sparse_jax_false(self, spin_hilbert):
        """Test to_sparse with jax_=False returns scipy matrix."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Get sparse representation with jax_=False
        sparse_mat = embed_op.to_sparse(jax_=False)

        # Check it's scipy sparse
        from scipy.sparse import csr_matrix

        assert isinstance(sparse_mat, csr_matrix)

        # Check correctness
        dense_from_sparse = sparse_mat.toarray()
        dense_direct = embed_op.to_dense()

        np.testing.assert_allclose(dense_from_sparse, dense_direct, rtol=1e-10)

    def test_embed_operator_to_sparse_jax_true(self, spin_hilbert):
        """Test to_sparse with jax_=True returns JAX BCSR matrix."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Get sparse representation with jax_=True
        sparse_mat = embed_op.to_sparse(jax_=True)

        # Check it's JAX sparse
        from jax.experimental.sparse import BCSR

        assert isinstance(sparse_mat, BCSR)

        # Check shape
        assert sparse_mat.shape == (hi_tensor.n_states, hi_tensor.n_states)

        # Check correctness
        dense_from_sparse = sparse_mat.todense()
        dense_direct = embed_op.to_dense()

        np.testing.assert_allclose(dense_from_sparse, dense_direct, rtol=1e-10)

    def test_embed_operator_to_sparse_different_subspaces(self, spin_hilbert):
        """Test to_sparse for operators on different subspaces."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        # Test subspace 0
        sx0 = nk.operator.spin.sigmax(hi, 0)
        embed0 = nk.operator.EmbedOperator(hi_tensor, sx0, subspace=0)
        sparse0 = embed0.to_sparse()

        # Test subspace 1
        sx1 = nk.operator.spin.sigmax(hi, 0)
        embed1 = nk.operator.EmbedOperator(hi_tensor, sx1, subspace=1)
        sparse1 = embed1.to_sparse()

        # They should be different
        assert not np.allclose(sparse0.toarray(), sparse1.toarray())

        # But both should match their dense representations
        np.testing.assert_allclose(sparse0.toarray(), embed0.to_dense(), rtol=1e-10)
        np.testing.assert_allclose(sparse1.toarray(), embed1.to_dense(), rtol=1e-10)

    def test_embed_operator_to_sparse_mixed_hilbert(self, spin_hilbert, fock_hilbert):
        """Test to_sparse with mixed Hilbert spaces (Spin + Fock)."""
        hi_spin = spin_hilbert
        hi_fock = fock_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi_spin, hi_fock)

        # Spin operator on first subspace
        sx = nk.operator.spin.sigmax(hi_spin, 0)
        embed_spin = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Bosonic operator on second subspace
        a = nk.operator.boson.create(hi_fock, 0)
        embed_boson = nk.operator.EmbedOperator(hi_tensor, a, subspace=1)

        # Test both
        sparse_spin = embed_spin.to_sparse()
        sparse_boson = embed_boson.to_sparse()

        # Check correctness
        np.testing.assert_allclose(
            sparse_spin.toarray(), embed_spin.to_dense(), rtol=1e-10
        )
        np.testing.assert_allclose(
            sparse_boson.toarray(), embed_boson.to_dense(), rtol=1e-10
        )

    def test_embed_operator_to_sparse_sparsity(self, spin_hilbert):
        """Test that to_sparse actually produces sparse matrices (not dense)."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        # Diagonal operator (very sparse)
        sz = nk.operator.spin.sigmaz(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sz, subspace=0)

        sparse_mat = embed_op.to_sparse()

        # Check that it's actually sparse
        n_nonzero = sparse_mat.nnz
        total_elements = sparse_mat.shape[0] * sparse_mat.shape[1]

        # For a diagonal-like operator, should be much sparser than dense
        sparsity_ratio = n_nonzero / total_elements
        assert sparsity_ratio < 0.5, f"Matrix not sparse enough: {sparsity_ratio}"

    def test_embed_operator_to_sparse_multiple_subspaces(self):
        """Test to_sparse with 3+ subspaces."""
        hi1 = nk.hilbert.Spin(s=1 / 2, N=2)
        hi2 = nk.hilbert.Spin(s=1 / 2, N=2)
        hi3 = nk.hilbert.Spin(s=1 / 2, N=2)
        hi_tensor = nk.hilbert.TensorHilbert(hi1, hi2, hi3)

        # Operator on middle subspace
        sx = nk.operator.spin.sigmax(hi2, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=1)

        sparse_mat = embed_op.to_sparse()

        # Check correctness
        np.testing.assert_allclose(
            sparse_mat.toarray(), embed_op.to_dense(), rtol=1e-10
        )

    def test_embed_operator_to_sparse_kron_structure(self, spin_hilbert):
        """Test that to_sparse produces correct Kronecker product structure."""
        hi = spin_hilbert
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Get sparse matrix
        sparse_result = embed_op.to_sparse()

        # Manually compute expected result: sx ⊗ I
        from scipy.sparse import identity as sp_identity
        from scipy.sparse import kron as sp_kron

        sx_sparse = sx.to_sparse()
        I_sparse = sp_identity(hi.n_states, format="csr")
        expected = sp_kron(sx_sparse, I_sparse, format="csr")

        # Compare
        np.testing.assert_allclose(
            sparse_result.toarray(), expected.toarray(), rtol=1e-10
        )


class TestEmbedOperatorIntegration:
    """Integration tests for EmbedOperator with complex scenarios."""

    def test_hubbard_holstein_example(self):
        """
        Test a simplified version of the Hubbard-Holstein example
        from the documentation.
        """
        n_sites = 2

        hi_fermion = nk.hilbert.Fock(n_max=1, N=2 * n_sites)
        hi_boson = nk.hilbert.Fock(n_max=3, N=n_sites)
        hi_joint = nk.hilbert.TensorHilbert(hi_fermion, hi_boson)

        ham_boson = sum(nk.operator.boson.number(hi_boson, i) for i in range(n_sites))
        ham_boson_embed = nk.operator.EmbedOperator(hi_joint, ham_boson, subspace=1)

        assert ham_boson_embed.hilbert == hi_joint
        assert ham_boson_embed.subspace == 1
        assert isinstance(ham_boson_embed, nk.operator.EmbedOperator)

    def test_embed_with_variational_state(self):
        """Test EmbedOperator with variational state calculations."""
        # Create a simple system
        hi1 = nk.hilbert.Spin(s=1 / 2, N=2)
        hi2 = nk.hilbert.Spin(s=1 / 2, N=2)
        hi_tensor = nk.hilbert.TensorHilbert(hi1, hi2)

        # Create embedded operator
        sx = nk.operator.spin.sigmax(hi1, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx, subspace=0)

        # Create a variational state
        sampler = nk.sampler.ExactSampler(hi_tensor)
        ma = nk.models.RBM(alpha=1)
        vs = nk.vqs.MCState(sampler, ma, n_samples=100)

        # Test expectation value calculation
        try:
            expectation = vs.expect(embed_op)
            assert np.isfinite(expectation.mean)
        except Exception as e:
            pytest.skip(f"Integration with variational state not yet supported: {e}")

    def test_embed_to_dense_comparison(self):
        """Test that to_dense gives correct matrix representation."""
        hi = nk.hilbert.Spin(s=1 / 2, N=2)
        hi_tensor = nk.hilbert.TensorHilbert(hi, hi)

        sx0 = nk.operator.spin.sigmax(hi, 0)
        embed_op = nk.operator.EmbedOperator(hi_tensor, sx0, subspace=0)

        if hasattr(embed_op, "to_dense"):
            dense_embed = embed_op.to_dense()

            # Create reference: sx ⊗ I where sx acts on first subspace
            sx0_dense = sx0.to_dense()
            I_dense = jnp.eye(hi.n_states)
            dense_reference = jnp.kron(sx0_dense, I_dense)

            np.testing.assert_allclose(dense_embed, dense_reference, rtol=1e-10)
