import pytest

import jax.numpy as jnp

import netket as nk

representation_list = []

# Qubit Hilbert space
hilbert_space = nk.hilbert.Qubit(3)
graph = nk.graph.Chain(3, pbc=True)
s3_representation = nk.symmetry.canonical_representation(
    hilbert_space, graph.space_group()
)
representation_list.append(pytest.param(s3_representation, id="S3"))

# Fermion Hilbert space
chain = nk.graph.Chain(4, pbc=True)
graph = nk.graph.disjoint_union(chain, chain)
hilbert_space_fermion = nk.hilbert.SpinOrbitalFermions(
    4, s=1 / 2, n_fermions_per_spin=(2, 2)
)
group = graph.automorphisms()

rep_dict = {
    g: nk.operator.permutation.PermutationOperatorFermion(hilbert_space_fermion, g)
    for g in group.elems
}
fermion_representation = nk.symmetry.Representation(group, rep_dict)
representation_list.append(pytest.param(fermion_representation, id="fermions"))

spin_flip_spin_representation = nk.symmetry.spin_flip_representation(
    nk.hilbert.Spin(1 / 2, 4)
)
representation_list.append(
    pytest.param(spin_flip_spin_representation, id="spin-flip-spin")
)

spin_flip_fermion_representation = nk.symmetry.spin_flip_representation(
    nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(2, 2))
)
representation_list.append(
    pytest.param(spin_flip_fermion_representation, id="spin-flip-fermion")
)


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
    space_group_representation_fermion = nk.symmetry.canonical_representation(
        hilbert, graph.space_group()
    )

    rep_dict = space_group_representation_fermion.representation_dict
    for op in rep_dict.values():
        assert len(op.permutation.permutation_array) == hilbert.size


def test_fermion_canonical_representation_with_duplicates():
    """
    Regression test for issue where physical_to_logical_permutation_group
    created duplicate permutations for fermionic Hilbert spaces, causing
    canonical_representation to fail with ValueError.

    This test uses a small 2x2 hypercube which triggers the duplicate
    permutation issue: the space group has 32 elements but after fermionic
    transformation, only 8 unique permutations remain.
    """
    # Create a small 2x2 hypercube graph
    g = nk.graph.Hypercube(length=2, n_dim=2, pbc=True)
    n_sites = g.n_nodes

    # Create a fermionic Hilbert space with spin-1/2 particles
    hi = nk.hilbert.SpinOrbitalFermions(n_sites, s=1 / 2, n_fermions_per_spin=(2, 2))

    # Get the space group
    sg = g.space_group()

    # This should not raise ValueError about mismatched dictionary size
    sg_rep = nk.symmetry.canonical_representation(hi, sg)

    # Verify the representation was constructed correctly
    assert len(sg_rep.representation_dict) == len(sg_rep.group)
    assert sg_rep.hilbert == hi

    # Verify all operators have the correct size
    for op in sg_rep.representation_dict.values():
        assert op.hilbert == hi


def test_spin_flip_action_on_spin():
    hilbert = nk.hilbert.Spin(1 / 2, 4)
    representation = nk.symmetry.spin_flip_representation(hilbert)
    spin_flip = representation[1]

    state = jnp.array([[1, -1, 1, -1]])
    new_state, matrix_elements = spin_flip.get_conn_padded(state)

    assert jnp.all(new_state == -state[:, None, :])
    assert jnp.all(matrix_elements == 1)


def test_spin_flip_action_on_fermions():
    hilbert = nk.hilbert.SpinOrbitalFermions(3, s=1 / 2, n_fermions_per_spin=(1, 1))
    representation = nk.symmetry.spin_flip_representation(hilbert)
    spin_flip = representation[1]

    state = jnp.array([[1, 0, 0, 0, 1, 0]])
    new_state, matrix_elements = spin_flip.get_conn_padded(state)

    assert jnp.all(new_state == jnp.array([[[0, 1, 0, 1, 0, 0]]]))
    assert jnp.all(matrix_elements == jnp.array([[-1]]))


def test_spin_flip_invalid_spin_constraint():
    hilbert = nk.hilbert.Spin(1 / 2, 4, total_sz=1)

    with pytest.raises(ValueError, match="zero total magnetization"):
        nk.symmetry.spin_flip_representation(hilbert)


def test_spin_flip_invalid_fermion_constraints():
    hilbert = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(2, 1))

    with pytest.raises(ValueError, match="does not preserve"):
        nk.symmetry.spin_flip_representation(hilbert)


def test_spin_flip_invalid_spinless_fermions():
    hilbert = nk.hilbert.SpinOrbitalFermions(4, n_fermions=2)

    with pytest.raises(ValueError, match="spinful"):
        nk.symmetry.spin_flip_representation(hilbert)
