import pytest

import jax.numpy as jnp

import netket as nk

representation_list = []

# Qubit Hilbert space
hilbert_space = nk.hilbert.Qubit(3)
graph = nk.graph.Chain(3, pbc=True)
s3_representation = graph.space_group_representation(hilbert_space)
representation_list.append(pytest.param(s3_representation, id="S3"))

# Fermion Hilbert space
chain = nk.graph.Chain(4, pbc=True)
graph = nk.graph.disjoint_union(chain, chain)
hilbert_space_fermion = nk.hilbert.SpinOrbitalFermions(
    4, s=1 / 2, n_fermions_per_spin=(2, 2)
)
group = (
    graph.automorphisms()
)  # The test could be made lighter by removing some symmetries
# For example by taking the product of point group of one chain with the translation group
# of the other

rep_dict = {
    g: nk.operator.permutation.PermutationOperatorFermion(hilbert_space_fermion, g)
    for g in group.elems
}
fermion_representation = nk.symmetry.Representation(group, rep_dict)
representation_list.append(pytest.param(fermion_representation, id="fermions"))


@pytest.mark.parametrize("representation", representation_list)
def test_projectors(representation):

    hilbert_space = representation.hilbert

    projector_list = [
        representation.projector(k)
        for k in range(representation.group.character_table().shape[0])
    ]
    projector_dense_list = [projector.to_dense() for projector in projector_list]

    # Check that the projectors are projectors
    for projector_dense in projector_dense_list:
        assert (
            jnp.linalg.norm(projector_dense @ projector_dense - projector_dense)
            / projector_dense.size
            < 1e-14
        )

    # Check that the projectors sum to identity
    projector_sum = sum(projector_dense_list)
    assert (
        jnp.linalg.norm(projector_sum - jnp.eye(hilbert_space.n_states))
        / projector_dense.size
        < 1e-14
    )


@pytest.mark.parametrize("representation", representation_list)
def test_irrep_dims(representation):

    irrep_dims_1 = representation.irrep_subspace_dims()
    _, irrep_dims_2 = representation.symmetry_adapted_basis()
    assert jnp.all(irrep_dims_1 == irrep_dims_2)


hilbert_list = [
    nk.hilbert.SpinOrbitalFermions(16, s=1 / 2, n_fermions_per_spin=(2, 2)),
    nk.hilbert.SpinOrbitalFermions(16, s=3 / 2, n_fermions_per_spin=(2, 2, 2, 2)),
]
graph_list = [
    nk.graph.Chain(16, pbc=True),
    nk.graph.Square(4, pbc=True),
    nk.graph.Triangular(extent=(4, 4), pbc=(True, True)),
]


@pytest.mark.parametrize("hilbert", hilbert_list)
@pytest.mark.parametrize("graph", graph_list)
def test_fermion_group_construction(
    hilbert: nk.hilbert.SpinOrbitalFermions, graph: nk.graph.Lattice
):
    space_group_representation_fermion = graph.space_group_representation(hilbert)

    rep_dict = space_group_representation_fermion.representation_dict
    for op in rep_dict.values():
        assert len(op.permutation.permutation_array) == hilbert.size
