import pytest

import jax
import jax.numpy as jnp
from flax import nnx

import netket as nk


# Common setup
hilbert_space = nk.hilbert.Qubit(3)
graph = nk.graph.Chain(3, pbc=True)
s3_representation = nk.symmetry.canonical_representation(
    hilbert_space, graph.space_group()
)
projector = s3_representation.projector(2)

ising_ham = nk.operator.IsingJax(
    hilbert_space, nk.graph.Chain(hilbert_space.size), 1, 1
)

operator_list = [ising_ham, projector, s3_representation[2]]


# Linen models and vstates
model_linen = nk.models.RBM(alpha=2)
sampler = nk.sampler.MetropolisLocal(hilbert_space)

mc_vstate_linen = nk.vqs.MCState(
    sampler, model_linen, n_samples=2**12, n_discard_per_chain=15
)
fs_vstate_linen = nk.vqs.FullSumState(hilbert_space, model_linen)

vstate_list_linen = [mc_vstate_linen, fs_vstate_linen]


# NNX model class
class RBMNNX(nnx.Module):
    """Simple NNX RBM for testing"""

    def __init__(self, N, alpha, rngs: nnx.Rngs, param_dtype=complex):
        self.linear = nnx.Linear(N, alpha * N, param_dtype=param_dtype, rngs=rngs)
        self.visible_bias = nnx.Param(
            jax.random.uniform(rngs.params(), (N,), dtype=param_dtype)
        )

    def __call__(self, x_in):
        y = nk.nn.activation.log_cosh(self.linear(x_in))
        y = jnp.sum(y, axis=-1)
        y = y + jnp.dot(x_in, self.visible_bias.value)
        return y


# NNX models and vstates
model_nnx = RBMNNX(hilbert_space.size, alpha=2, param_dtype=float, rngs=nnx.Rngs(0))

mc_vstate_nnx = nk.vqs.MCState(
    sampler, model_nnx, n_samples=2**12, n_discard_per_chain=15
)
fs_vstate_nnx = nk.vqs.FullSumState(hilbert_space, model_nnx)

vstate_list_nnx = [mc_vstate_nnx, fs_vstate_nnx]


@pytest.mark.parametrize("operator", operator_list)
@pytest.mark.parametrize("vstate", vstate_list_linen)
def test_apply_operator_linen(operator, vstate):
    """Test apply_operator with Linen modules"""

    transformed_vstate = nk.vqs.apply_operator(operator, vstate)

    transformed_vstate_dense_1 = transformed_vstate.to_array(normalize=False)
    transformed_vstate_dense_2 = operator.to_dense() @ vstate.to_array(normalize=False)

    assert (
        jnp.linalg.norm(transformed_vstate_dense_1 - transformed_vstate_dense_2) < 1e-14
    )

    assert transformed_vstate.hilbert == vstate.hilbert

    if isinstance(vstate, nk.vqs.FullSumState):
        assert isinstance(transformed_vstate, nk.vqs.FullSumState)

    if isinstance(vstate, nk.vqs.MCState):
        assert isinstance(transformed_vstate, nk.vqs.MCState)
        assert transformed_vstate.sampler == vstate.sampler
        assert transformed_vstate.n_samples == vstate.n_samples
        assert transformed_vstate.n_samples_per_rank == vstate.n_samples_per_rank
        assert transformed_vstate.n_discard_per_chain == vstate.n_discard_per_chain

    # Check that the transformed model is an ApplyOperatorModuleLinen
    from netket._src.nn.apply_operator.linen import ApplyOperatorModuleLinen

    assert isinstance(transformed_vstate._model, ApplyOperatorModuleLinen)

    # change the chunk size and check that it has been adapted correctly.
    vstate.chunk_size = 2**10
    transformed_vstate = nk.vqs.apply_operator(operator, vstate)

    assert transformed_vstate.chunk_size == vstate.chunk_size // operator.max_conn_size


@pytest.mark.parametrize("operator", operator_list)
@pytest.mark.parametrize("vstate", vstate_list_nnx)
def test_apply_operator_nnx(operator, vstate):
    """Test apply_operator with NNX modules"""

    transformed_vstate = nk.vqs.apply_operator(operator, vstate)

    # Check that the transformed state computes the correct result
    transformed_vstate_dense_1 = transformed_vstate.to_array(normalize=False)
    transformed_vstate_dense_2 = operator.to_dense() @ vstate.to_array(normalize=False)

    assert (
        jnp.linalg.norm(transformed_vstate_dense_1 - transformed_vstate_dense_2) < 1e-12
    )

    assert transformed_vstate.hilbert == vstate.hilbert

    # Check that the correct type is preserved
    if isinstance(vstate, nk.vqs.FullSumState):
        assert isinstance(transformed_vstate, nk.vqs.FullSumState)

    if isinstance(vstate, nk.vqs.MCState):
        assert isinstance(transformed_vstate, nk.vqs.MCState)
        assert transformed_vstate.sampler == vstate.sampler
        assert transformed_vstate.n_samples == vstate.n_samples
        assert transformed_vstate.n_samples_per_rank == vstate.n_samples_per_rank
        assert transformed_vstate.n_discard_per_chain == vstate.n_discard_per_chain

    # Check that the transformed model is an ApplyOperatorModuleNNX
    from netket._src.nn.apply_operator.nnx import ApplyOperatorModuleNNX

    assert isinstance(transformed_vstate.model, ApplyOperatorModuleNNX)
    # Check that we can access the operator via the property
    assert type(transformed_vstate.model.operator) == type(operator)

    # Check chunk size adaptation
    vstate.chunk_size = 2**10
    transformed_vstate = nk.vqs.apply_operator(operator, vstate)
    assert transformed_vstate.chunk_size == vstate.chunk_size // operator.max_conn_size


@pytest.mark.parametrize("vstate", vstate_list_linen + vstate_list_nnx)
def test_apply_operator_twice(vstate):
    """Test applying operator twice"""
    transformed_vstate_1 = nk.vqs.apply_operator(ising_ham, vstate)
    transformed_vstate_2 = nk.vqs.apply_operator(projector, transformed_vstate_1)

    result_1 = transformed_vstate_2.to_array(normalize=False)
    result_2 = (
        projector.to_dense() @ ising_ham.to_dense() @ vstate.to_array(normalize=False)
    )

    assert jnp.linalg.norm(result_1 - result_2) < 1e-12
