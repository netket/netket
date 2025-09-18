import pytest

import jax.numpy as jnp

import netket as nk


hilbert_space = nk.hilbert.Qubit(3)
graph = nk.graph.Chain(3, pbc=True)
s3_representation = graph.space_group_representation(hilbert_space)
projector = s3_representation.projector(2)

ising_ham = nk.operator.IsingJax(
    hilbert_space, nk.graph.Chain(hilbert_space.size), 1, 1
)

operator_list = [ising_ham, projector, s3_representation[2]]


model = nk.models.RBM(alpha=2)
sampler = nk.sampler.MetropolisLocal(hilbert_space)

mc_vstate = nk.vqs.MCState(sampler, model, n_samples=2**12, n_discard_per_chain=15)
fs_vstate = nk.vqs.FullSumState(hilbert_space, model)

vstate_list = [mc_vstate, fs_vstate]


@pytest.mark.parametrize("operator", operator_list)
@pytest.mark.parametrize("vstate", vstate_list)
def test_apply_operator(operator, vstate):

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

    # change the chunk size and check that it has been adapted correctly.
    vstate.chunk_size = 2**10
    transformed_vstate = nk.vqs.apply_operator(operator, vstate)

    assert transformed_vstate.chunk_size == vstate.chunk_size // operator.max_conn_size
