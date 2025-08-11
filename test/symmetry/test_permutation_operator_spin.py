import pytest

import jax.numpy as jnp

import netket as nk

from itertools import product


def test_permutation_operator():

    translation_1 = nk.utils.group.Permutation(
        permutation_array=jnp.array([1, 2, 0]), name="translation_1"
    )
    translation_2 = nk.utils.group.Permutation(
        permutation_array=jnp.array([2, 0, 1]), name="translation_2"
    )
    transposition_12 = nk.utils.group.Permutation(
        permutation_array=jnp.array([0, 2, 1]), name="transposition_12"
    )
    identity = nk.utils.group.Identity()

    hilbert_space = nk.hilbert.Qubit(3)
    spin_1_space = nk.hilbert.Spin(1, 3)
    hilbert_space_4_qubits = nk.hilbert.Qubit(4)

    translation_1_operator = nk.symmetry.PermutationOperator(
        hilbert_space, translation_1
    )
    translation_2_operator = nk.symmetry.PermutationOperator(
        hilbert_space, translation_2
    )
    transposition_12_operator = nk.symmetry.PermutationOperator(
        hilbert_space, transposition_12
    )
    identity_operator = nk.symmetry.PermutationOperator(hilbert_space, identity)
    permutation_operators = (
        translation_1_operator,
        translation_2_operator,
        transposition_12_operator,
        identity_operator,
    )

    translation_1_prime = nk.utils.group.Permutation(
        permutation_array=jnp.array([1, 2, 0]), name="translation_1"
    )

    translation_1_operator_spin_1 = nk.symmetry.PermutationOperator(
        spin_1_space, translation_1
    )

    with pytest.raises(ValueError):
        nk.symmetry.PermutationOperator(hilbert_space_4_qubits, translation_1)

    with pytest.raises(TypeError):
        nk.symmetry.PermutationOperator(hilbert_space, jnp.array([0, 1, 2]))

    # Check equality works properly
    assert translation_1 == translation_1_prime
    assert not translation_1 == translation_2

    # Check equality depends on Hilbert space
    assert not translation_1 == translation_1_operator_spin_1

    # Check representation property
    for op_1, op_2 in product(permutation_operators, permutation_operators):
        product_permutation = op_1.permutation @ op_2.permutation
        product_permutation_operator = nk.symmetry.PermutationOperator(
            op_1.hilbert, product_permutation
        )
        product_permutation_dense = product_permutation_operator.to_dense()
        assert (
            jnp.linalg.norm(
                product_permutation_dense - op_1.to_dense() @ op_2.to_dense()
            )
            < 1e-14
        )

        # Check product
        assert (
            jnp.linalg.norm(
                (op_1 @ op_2).to_dense() - op_1.to_dense() @ op_2.to_dense()
            )
        ) < 1e-14

    # Check to_dense
    translation_1_dense_ref = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    transposition_12_dense_ref = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    assert (
        jnp.linalg.norm(translation_1_operator.to_dense() - translation_1_dense_ref)
        < 1e-14
    )
    assert (
        jnp.linalg.norm(
            transposition_12_operator.to_dense() - transposition_12_dense_ref
        )
        < 1e-14
    )
    assert (
        jnp.linalg.norm(identity_operator.to_dense() - jnp.eye(hilbert_space.n_states))
        < 1e-14
    )


op_list = []

graph = nk.graph.Chain(3, pbc=True)
hilbert_space = nk.hilbert.Qubit(3)
for permutation in graph.space_group().elems:
    op = nk.symmetry.PermutationOperator(hilbert_space, permutation)
    op_list.append(op)

graph = nk.graph.Square(2, pbc=False)
hilbert_space = nk.hilbert.Spin(1, 4)
for permutation in graph.space_group().elems:
    op = nk.symmetry.PermutationOperator(hilbert_space, permutation)
    op_list.append(op)


@pytest.mark.parametrize("op", op_list)
def test_trace(op):
    assert jnp.trace(op.to_dense()).item() == op.trace()
