import pytest

import jax.numpy as jnp

import netket as nk


hilbert_space = nk.hilbert.Qubit(3)
graph = nk.graph.Chain(3, pbc=True)
group = graph.space_group()

rep_dict = {g: nk.symmetry.PermutationOperator(hilbert_space, g) for g in group.elems}
s3_representation = nk.symmetry.Representation(group, rep_dict)

representation_list = [s3_representation]


@pytest.mark.parametrize("representation", representation_list)
def test_representation(representation):

    hilbert_space = representation.hilbert_space

    projector_list = [
        representation.get_projector(k)
        for k in range(representation.group.character_table().shape[0])
    ]
    projector_dense_list = [projector.to_dense() for projector in projector_list]

    # Check that the projectors are projectors
    for projector_dense in projector_dense_list:
        assert (
            jnp.linalg.norm(projector_dense @ projector_dense - projector_dense) < 1e-14
        )

    # Check that the projectors sum to identity
    projector_sum = sum(projector_dense_list)
    assert jnp.linalg.norm(projector_sum - jnp.eye(hilbert_space.n_states)) < 1e-14
