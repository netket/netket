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

import pytest
import jax
import jax.numpy as jnp

import netket as nk
import netket.experimental as nkx
from netket.operator import fermion


class TestDeterminantVariationalState:
    @pytest.fixture
    def small_hilbert(self):
        """Small fermionic system for testing."""
        return nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)

    @pytest.fixture
    def spin_hilbert(self):
        """Spin-1/2 fermionic system."""
        return nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(1, 1))

    def test_initialization_generalized(self, small_hilbert):
        """Test initialization with generalized HF."""
        vstate = nkx.vqs.DeterminantVariationalState(small_hilbert, generalized=True)

        assert vstate.hilbert == small_hilbert
        assert vstate.n_parameters > 0
        assert isinstance(vstate.model, nk.models.Slater2nd)

    def test_invalid_hilbert(self):
        """Test that non-SpinOrbitalFermions hilbert raises error."""
        hi = nk.hilbert.Spin(0.5, 4)

        with pytest.raises(TypeError, match="SpinOrbitalFermions"):
            nkx.vqs.DeterminantVariationalState(hi)

        """Test that hilbert without fixed particle number raises error."""
        hi = nk.hilbert.SpinOrbitalFermions(4)  # No n_fermions specified

        # DeterminantVariationalState will raise an error for no fixed n_fermions
        with pytest.raises(ValueError, match="fixed number"):
            nkx.vqs.DeterminantVariationalState(hi, generalized=True)

    def test_log_value(self, small_hilbert):
        """Test log_value evaluation."""
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, seed=123
        )

        # Create a few sample configurations
        samples = small_hilbert.random_state(jax.random.PRNGKey(456), 10)
        log_vals = vstate.log_value(samples)

        assert log_vals.shape == (10,)
        assert jnp.all(jnp.isfinite(log_vals))

    def test_rdm_shape_generalized(self, small_hilbert):
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, seed=123
        )
        assert vstate.rdm.shape == (small_hilbert.size, small_hilbert.size)
        assert jnp.allclose(vstate.rdm, vstate.rdm.conj().T)  # is hermitian

    def test_rdm_shape_restricted(self, spin_hilbert):
        vstate = nkx.vqs.DeterminantVariationalState(
            spin_hilbert, generalized=False, restricted=True, seed=123
        )
        assert vstate.rdm.shape == (spin_hilbert.size, spin_hilbert.size)
        assert jnp.allclose(vstate.rdm, vstate.rdm.conj().T)

    def test_rdm_caching(self, small_hilbert):
        """Test that density matrix is cached and cleared on parameter update."""
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, seed=123
        )

        # First access - should compute
        rdm1 = vstate.rdm
        # Second access - should use cache
        rdm2 = vstate.rdm
        assert rdm1 is rdm2

        # After parameter update, cache should be cleared
        vstate.init_parameters(seed=456)
        rdm3 = vstate.rdm
        assert rdm3 is not rdm1

    @pytest.fixture
    def operators(self, small_hilbert):
        """Fixture providing test operators."""
        return {
            "number_total": sum(
                fermion.number(small_hilbert, i) for i in range(small_hilbert.size)
            ).to_normal_order(),
            "number_site0": fermion.number(small_hilbert, 0).to_normal_order(),
            "hopping": (
                fermion.create(small_hilbert, 0) @ fermion.destroy(small_hilbert, 1)
                + fermion.create(small_hilbert, 1) @ fermion.destroy(small_hilbert, 0)
            ).to_normal_order(),
        }

    @pytest.mark.parametrize("op_name", ["number_total", "number_site0", "hopping"])
    def test_expectation(self, small_hilbert, operators, op_name):
        """Test expectation values match exact calculation."""
        vstate_mf = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, param_dtype=complex, seed=123
        )
        vstate_exact = vstate_mf.to_fullsumstate()

        op = operators[op_name]

        exp_mf = vstate_mf.expect(op)
        exp_exact = vstate_exact.expect(op)

        assert jnp.allclose(exp_mf.mean, exp_exact.mean, atol=1e-10)
        assert exp_mf.variance == 0.0  # Deterministic

    @pytest.mark.parametrize("op_name", ["number_total", "number_site0", "hopping"])
    def test_expect_and_grad(self, small_hilbert, operators, op_name):
        """Test expect_and_grad matches exact calculation."""
        vstate_mf = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, param_dtype=complex, seed=123
        )
        vstate_exact = vstate_mf.to_fullsumstate()

        op = operators[op_name]

        exp_mf, grad_mf = vstate_mf.expect_and_grad(op)
        exp_exact, grad_exact = vstate_exact.expect_and_grad(op)

        assert jnp.allclose(exp_mf.mean, exp_exact.mean, atol=1e-10)

        # Check gradients match (conjugate due to holomorphic=False convention)
        def compare_grads(g_mf, g_exact):
            assert jnp.allclose(g_mf, jnp.conj(g_exact), atol=1e-10)

        jax.tree_util.tree_map(compare_grads, grad_mf, grad_exact)

    @pytest.mark.parametrize("op_name", ["number_total", "number_site0", "hopping"])
    def test_expect_and_forces(self, small_hilbert, operators, op_name):
        """Test expect_and_forces matches exact calculation."""
        vstate_mf = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, param_dtype=complex, seed=123
        )
        vstate_exact = vstate_mf.to_fullsumstate()

        op = operators[op_name]

        exp_mf, forces_mf = vstate_mf.expect_and_forces(op)
        exp_exact, forces_exact = vstate_exact.expect_and_forces(op)

        assert jnp.allclose(exp_mf.mean, exp_exact.mean, atol=1e-10)

        # For deterministic states, forces = gradients
        # For stochastic states, forces have different scaling
        # Check forces match (with conjugate and 2x scaling for FullSumState)
        def compare_forces(f_mf, f_exact):
            assert jnp.allclose(f_mf, 2 * jnp.conj(f_exact), atol=1e-10)

        jax.tree_util.tree_map(compare_forces, forces_mf, forces_exact)

    def test_qgt_raises_error(self, small_hilbert):
        """Test that QGT raises NotImplementedError."""
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, seed=123
        )

        with pytest.raises(NotImplementedError, match="Quantum Geometric Tensor"):
            vstate.quantum_geometric_tensor()

    def test_to_fullsumstate(self, small_hilbert):
        """Test conversion to FullSumState."""
        vstate_mf = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, param_dtype=complex, seed=123
        )

        vstate_full = vstate_mf.to_fullsumstate()

        assert isinstance(vstate_full, nk.vqs.FullSumState)
        assert vstate_full.hilbert == vstate_mf.hilbert

        # Parameters should be preserved
        # Check a few sample points
        samples = small_hilbert.random_state(jax.random.PRNGKey(456), 5)
        log_vals_mf = vstate_mf.log_value(samples)
        log_vals_full = vstate_full.log_value(samples)

        assert jnp.allclose(log_vals_mf, log_vals_full)

    def test_to_mcstate(self, small_hilbert):
        """Test conversion to MCState."""
        vstate_mf = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, param_dtype=complex, seed=123
        )

        # Create a simple sampler
        sampler = nk.sampler.MetropolisLocal(small_hilbert)

        vstate_mc = vstate_mf.to_mcstate(sampler, n_samples=100)

        assert isinstance(vstate_mc, nk.vqs.MCState)
        assert vstate_mc.hilbert == vstate_mf.hilbert
        assert vstate_mc.sampler == sampler

        # Parameters should be preserved
        samples = small_hilbert.random_state(jax.random.PRNGKey(456), 5)
        log_vals_mf = vstate_mf.log_value(samples)
        log_vals_mc = vstate_mc.log_value(samples)

        assert jnp.allclose(log_vals_mf, log_vals_mc)

    def test_to_mcstate_invalid_sampler(self, small_hilbert):
        """Test that to_mcstate with wrong hilbert raises error."""
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, seed=123
        )

        # Create sampler with different hilbert
        wrong_hilbert = nk.hilbert.SpinOrbitalFermions(6, n_fermions=3)
        sampler = nk.sampler.MetropolisLocal(wrong_hilbert)

        with pytest.raises(ValueError, match="hilbert space"):
            vstate.to_mcstate(sampler, n_samples=100)

    def test_to_array(self, small_hilbert):
        """Test to_array method."""
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, param_dtype=complex, seed=123
        )

        psi = vstate.to_array(normalize=True)

        # Check shape
        assert psi.shape == (small_hilbert.n_states,)

        # Check normalization
        assert jnp.allclose(jnp.sum(jnp.abs(psi) ** 2), 1.0, atol=1e-6)

    def test_serialization(self, small_hilbert, tmp_path):
        """Test serialization and deserialization."""
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, seed=123
        )

        # Compute something
        op = fermion.number(small_hilbert, 0).to_normal_order()
        exp_before = vstate.expect(op).mean

        # Save
        import flax

        state_dict = flax.serialization.to_state_dict(vstate)

        # Create new state and load
        vstate_new = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, seed=456
        )
        vstate_new = flax.serialization.from_state_dict(vstate_new, state_dict)

        # Should get same expectation
        exp_after = vstate_new.expect(op).mean

        assert jnp.allclose(exp_before, exp_after)

    def test_parameter_update_clears_cache(self, small_hilbert):
        """Test that setting parameters clears the density matrix cache."""
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, seed=123
        )

        # Access density matrix to cache it
        _ = vstate.rdm
        assert vstate._rdm is not None

        # Update parameters
        new_params = vstate.parameters  # Get a copy
        vstate.parameters = new_params

        # Cache should be cleared
        assert vstate._rdm is None

    def test_training_simple(self, small_hilbert):
        """Test that the state can be used in a simple training loop."""
        vstate = nkx.vqs.DeterminantVariationalState(
            small_hilbert, generalized=True, param_dtype=complex, seed=123
        )

        # Simple Hamiltonian: kinetic energy
        ham = sum(
            fermion.create(small_hilbert, i) @ fermion.destroy(small_hilbert, j)
            + fermion.create(small_hilbert, j) @ fermion.destroy(small_hilbert, i)
            for i in range(small_hilbert.size - 1)
            for j in range(i + 1, small_hilbert.size)
        )
        ham = ham.to_normal_order()

        # Get initial energy
        e0 = vstate.expect(ham).mean

        # Do a gradient step
        _, grad = vstate.expect_and_grad(ham)

        # Update parameters
        lr = 0.01
        new_params = jax.tree_util.tree_map(
            lambda p, g: p - lr * g, vstate.parameters, grad
        )
        vstate.parameters = new_params

        # Get new energy
        e1 = vstate.expect(ham).mean

        # Energy should have changed
        assert e0 != e1
