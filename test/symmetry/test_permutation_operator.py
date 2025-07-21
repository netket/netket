import pytest

import jax
import jax.numpy as jnp

import netket as nk

from netket.utils.group import Permutation


def test_permutation_operator():

    translation_1 = Permutation(
        permutation_array=jnp.array([1, 2, 0]), name="translation_1"
    )
    translation_2 = Permutation(
        permutation_array=jnp.array([2, 0, 1]), name="translation_2"
    )
    cycle_12 = Permutation(permutation_array=jnp.array([0, 2, 1]), name="cycle_12")

    hilbert_space = nk.hilbert.Qubit(3)
    spin_1_space = nk.hilbert.Spin(1, 3)
    hilbert_space_4_qubits = nk.hilbert.Qubit(4)

    translation_1_operator = nk.symmetry.PermutationOperator(
        hilbert_space, translation_1
    )
    translation_2_operator = nk.symmetry.PermutationOperator(
        hilbert_space, translation_2
    )
    cycle_12_operator = nk.symmetry.PermutationOperator(hilbert_space, cycle_12)

    translation_1_prime = Permutation(
        permutation_array=jnp.array([1, 2, 0]), name="translation_1"
    )

    translation_1_operator_spin_1 = nk.symmetry.PermutationOperator(
        spin_1_space, translation_1
    )

    with pytest.raises(AssertionError):
        nk.symmetry.PermutationOperator(hilbert_space_4_qubits, translation_1)

    with pytest.raises(AssertionError):
        nk.symmetry.PermutationOperator(hilbert_space, jnp.array([0, 1, 2]))

    # Check equality works properly
    assert translation_1 == translation_1_prime
    assert not translation_1 == translation_2

    # Check equality depends on Hilbert space
    assert not translation_1 == translation_1_operator_spin_1

    pytree = (translation_1_operator, translation_2_operator, cycle_12_operator)
    inverse_pytree = (translation_2_operator, translation_1_operator, cycle_12_operator)

    # Check pytrification works properly
    assert not jax.tree.map(jnp.argsort, pytree) == pytree
    assert jax.tree.map(jnp.argsort, pytree) == inverse_pytree

    # Check to_dense
