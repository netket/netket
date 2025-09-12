import pytest

import jax.numpy as jnp
import jax

import netket as nk

from netket.operator.permutation import PermutationOperatorFermion
from netket.hilbert import SpinOrbitalFermions

from netket._src.operator.permutation.permutation_operator_fermion import (
    get_antisymmetric_signs,
)

seed = jax.random.PRNGKey(77)


def test_fermionic_sign():
    x = jnp.array(
        [
            [[0, 0, 1, 1, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1]],
            [[1, 0, 1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 1, 0, 0, 1]],
        ],
        dtype=int,
    )

    permutation_array = jnp.array([7, 6, 3, 2, 0, 1, 5, 4], dtype=int)

    signs = jnp.array([[-1, 1], [1, 1]], dtype=int)

    assert jnp.all(get_antisymmetric_signs(x, permutation_array, n_fermions=4) == signs)


def test_perm_op_fermion():

    key = jax.random.PRNGKey(0)
    hilbert = SpinOrbitalFermions(4, 1, n_fermions_per_spin=(2, 2, 3))
    x = hilbert.random_state(key, size=3)

    permutation_array = jnp.array([4, 6, 7, 5, 3, 1, 2, 0, 9, 10, 8, 11])

    permutation = nk.symmetry.group.Permutation(
        permutation_array=permutation_array, name="test_permutation"
    )

    perm_op = PermutationOperatorFermion(hilbert, permutation)
    x_primes, mels = perm_op.get_conn_padded(x)

    assert jnp.all(x_primes == x[..., jnp.newaxis, permutation_array])

    assert jnp.all(
        mels
        == get_antisymmetric_signs(
            x, jnp.argsort(permutation_array), hilbert.n_fermions
        )[..., jnp.newaxis]
    )


spin_1_hilbert_space = nk.hilbert.SpinOrbitalFermions(
    4, 1, n_fermions_per_spin=(2, 2, 3)
)
permutation_arrays = [
    jnp.array([7, 1, 9, 6, 8, 0, 3, 11, 4, 5, 2, 10]),
    jnp.array([6, 0, 5, 10, 3, 2, 4, 7, 9, 1, 8, 11]),
    jnp.array([6, 0, 5, 10, 3, 2, 4, 7, 9, 1, 8, 11]),
]


@pytest.mark.parametrize("permutation_array", permutation_arrays)
def test_invalid_permutations(permutation_array):
    permutation = nk.symmetry.group.Permutation(permutation_array=permutation_array)
    with pytest.raises(AssertionError):
        _ = PermutationOperatorFermion(spin_1_hilbert_space, permutation)


op_list = []


base_graph = nk.graph.Square(2, pbc=False)
graph = nk.graph.disjoint_union(base_graph, base_graph)
permutations = graph.automorphisms().elems

hilbert_space = nk.hilbert.SpinOrbitalFermions(4, 1 / 2, n_fermions_per_spin=(2, 2))

for permutation in permutations[::100]:
    op = PermutationOperatorFermion(hilbert_space, permutation)

    op_list.append(pytest.param(op, id="square_2_2"))


hilbert_space = nk.hilbert.SpinOrbitalFermions(4, 1 / 2, n_fermions_per_spin=(1, 3))

for permutation in permutations[: len(permutations) // 2 : 20]:
    op = PermutationOperatorFermion(hilbert_space, permutation)
    op_list.append(pytest.param(op, id="square_1_3"))


base_graph = nk.graph.Chain(3, pbc=True)
graph = nk.graph.disjoint_union(base_graph, base_graph, base_graph)
permutations = graph.automorphisms().elems

hilbert_space = nk.hilbert.SpinOrbitalFermions(3, 1, n_fermions_per_spin=(2, 2, 2))

for permutation in permutations[::100]:
    op = PermutationOperatorFermion(hilbert_space, permutation)
    op_list.append(pytest.param(op, id="chain_2_2_2"))


@pytest.mark.parametrize("op", op_list)
def test_trace(op):
    assert jnp.trace(op.to_dense()).item() == op.trace()
