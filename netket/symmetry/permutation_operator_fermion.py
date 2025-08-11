import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from itertools import product
from functools import partial

from netket.hilbert import SpinOrbitalFermions
from netket.utils.group import Permutation
from netket.symmetry import PermutationOperatorBase


def get_parity(array: jax.Array) -> jax.Array:
    """
    Count the parity of an array.
    This is the number of inversions in the array modulo 2.
    An inversion is a pair (i, j) such that i < j
    and array[i] > array[j].
    """
    batch_dims = array.shape[:-1]
    inversion_matrix = array[..., :, jnp.newaxis] > array[..., jnp.newaxis, :]
    upper_triangular_mask = jnp.triu(
        jnp.ones((*batch_dims, array.shape[-1], array.shape[-1]), dtype=bool), k=1
    )
    inversion_count = jnp.sum(inversion_matrix & upper_triangular_mask, axis=(-2, -1))
    return inversion_count % 2


def get_occupied_orbitals(x: jax.Array, n_fermions: int) -> jax.Array:
    """Return the indices of the occupied orbitals
    in a given SpinOrbitalFermions state n"""
    batch_dims, physical_dims = x.shape[:-1], x.shape[-1]
    x = x.reshape(-1, physical_dims)
    occupied_orbitals = _get_occupied_orbitals(x, n_fermions)
    return occupied_orbitals.reshape(*batch_dims, n_fermions)


@partial(jax.vmap, in_axes=(0, None))
def _get_occupied_orbitals(x: jax.Array, n_fermions: int) -> jax.Array:
    return x.nonzero(size=n_fermions)[0]


@partial(jax.jit, static_argnames=("n_fermions",))
def get_antisymmetric_signs(
    x: jax.Array, permutation: jax.Array, n_fermions: int
) -> jax.Array:
    """Return the sign of the permutation for a batch of fermionic Fock states x."""
    occupied = get_occupied_orbitals(x, n_fermions)
    permuted = permutation[occupied]
    parity = get_parity(permuted)
    sign = 1 - 2 * parity
    return sign


def get_subset_occupations(partition_labels, subsets):
    """
    Given a list of subsets of [0, ..., n-1] and a partition of [0, ..., n-1],
    return a list of the number of element in each partition for each subset.

    Args:
        partition_labels (<int>): The list such that partition_labels[k] is the partition to which k belongs.
        subsets (<<int>>): The list of subsets.

    Return:
        <ndarray>: The list such that l[i][j] is the number of elements of subsets[i] in partition j.
    """
    n_partitions = len(np.unique(partition_labels))
    occupations = []
    for subset in subsets:
        occupation_count = np.zeros(n_partitions, dtype=int)
        for k in subset:
            occupation_count[partition_labels[k]] += 1
        occupations.append(occupation_count)
    return occupations


def get_parity_sum(occupation_list, n_occupations):
    """
    Given a list occupation_list of lists of length p and n_occupations, a list of length p,
    we look at all subsets of occupation_list such that the sum of its elements is n_occupations.
    We return the sum over all such subsets, of the parity of the number of indices k in that subset such
    that sum(occupation_list[k]) is even.

    Args:
        occupation_list (<ndarray>): The list of lists of length p.
        n_occupations (<int>): The list of target occupation.

    Return:
        int: The sum of parities specified above.
    """

    table = np.full(
        (
            len(occupation_list) + 1,
            *(n_occupation + 1 for n_occupation in n_occupations),
        ),
        -1,
    )

    table[0] = 0
    table[0][(0,) * len(n_occupations)] = 1

    for i in range(len(occupation_list)):

        occupation_iterator = product(
            *list(range(n_occupation + 1) for n_occupation in n_occupations)
        )

        for occupation in occupation_iterator:

            not_included_sum = np.array(occupation) - occupation_list[i]
            if np.any(not_included_sum < 0):
                table[i + 1][occupation] = table[i][occupation]
            else:
                table[i + 1][occupation] = (-1) ** (
                    sum(occupation_list[i]) + 1
                ) * table[i][tuple(not_included_sum)] + table[i][occupation]

    return table[-1][n_occupations].item()


@register_pytree_node_class
class PermutationOperatorFermion(PermutationOperatorBase):
    """
    Permutation operator on a fermion space.
    ONLY WORKS FOR A HILBERT SPACE WITH FIXED NUMBER OF FERMIONS.
    For mathematical details on the definition of a permutation operator
    and its justification, we refer to [SOMEWHERE].

    Maybe we should also check that the operator is well-defined for the
    given Hilbert space. If the number of fermion per spin sector is fixed,
    we might want to check that the permutation respects that constraint.

    But then spin flip would be a permutation that is not a product of permutation
    acting on each subsector, but that is still valid. So it is hard to tell
    at a glance whether a given permutation is valid for restriction to that subspace.

    I don't think it is possible to make such a check. So we just have to hope
    the user knows what they are doing.

    Args:
        hilbert: The Hilbert space.
        permutation: The permutation represented by the operator.
    """

    def __init__(self, hilbert: SpinOrbitalFermions, permutation: Permutation):
        assert isinstance(hilbert, SpinOrbitalFermions)
        if hilbert.n_fermions is None:
            raise TypeError("The Hilbert space must have a fixed number of fermions.")
        super().__init__(hilbert, permutation)

    def __repr__(self):
        if self.permutation._name is not None:
            return f"PermutationOperatorFermion({self.permutation._name}: {self.permutation.permutation_array})"
        else:
            return f"PermutationOperatorFermion({self.permutation.permutation_array})"

    def __eq__(self, other):
        if isinstance(other, PermutationOperatorFermion):
            return (
                self.hilbert == other.hilbert and self.permutation == other.permutation
            )
        else:
            return False

    def get_signs(self, x):
        return get_antisymmetric_signs(
            x, self.permutation.inverse_permutation_array, self.hilbert.n_fermions
        )

    def get_conn_padded(self, x):
        r"""
        This function computes <x|Ug = <x o g| \xi_{g^{-1}}(x).
        where x is a batch of fermionic Fock states,
        x o g are the permuted occupation numbers and
        \xi_{g^{-1}}(x) is the sign of the permutation.
        """

        x = jnp.asarray(x)
        connected_elements = x.at[..., None, self.permutation.permutation_array].get(
            unique_indices=True, mode="promise_in_bounds"
        )
        signs = self.get_signs(x).astype(jnp.float32)
        return connected_elements, signs[..., jnp.newaxis]

    def __matmul__(self, other):
        if isinstance(other, PermutationOperatorFermion):
            return PermutationOperatorFermion(
                self.hilbert, self.permutation @ other.permutation
            )
        else:
            return super().__matmul__(other)

    def trace(self):
        partition_labels = sum(
            [
                self.hilbert.n_orbitals * [k]
                for k in range(self.hilbert.n_spin_subsectors)
            ],
            start=[],
        )
        cycle_decomposition = self.permutation.get_cycle_decomposition()
        cycle_occupation = get_subset_occupations(partition_labels, cycle_decomposition)
        return get_parity_sum(cycle_occupation, self.hilbert.n_fermions_per_spin)
