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

import warnings

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from functools import partial

from netket.hilbert import SpinOrbitalFermions
from netket.symmetry.group import Permutation
from netket.utils.types import DType

from netket._src.operator.permutation.permutation_operator_base import (
    PermutationOperatorBase,
)
from netket._src.operator.permutation.trace_utils import (
    get_subset_occupations,
    get_parity_sum,
)


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


def check_permutation_compatibility(
    permutation: Permutation, partition: list, constraints: list
):
    """
    Check that a permutation is compatible with constraints.

    Given a permutation that acts on n elements and a partition of those n elements
    that constrains bitstrings to contain exactly constraints[k] bits 1 in the
    partition subset partition[k], check that the permutation only maps compatible
    bitstrings to compatible bitstrings. In other words, it checks that the
    permutation operator defined by that permutation is well-defined on the
    Hilbert space specified by the constraints.

    It can be shown that it is equivalent to check that each partition subset
    is mapped by the inverse permutation onto another partition subset that has
    the same constraint. This equivalence only holds if the constraints are
    non-trivial, that is, only if the bitstrings are not constrained to be either
    only 0 or only 1 within any partition subset.

    Args:
        permutation: The permutation whose compatibility we check.
        partition: The partition that defines the constraint. It must be a partition of `range(n)` specified as a
            list where the k-th element is a list that specifies the elements of the k-th partition subset.
        constraints: The occupation constraint on each partition subset.
            The k-th constraint must be an integer contained strictly between 0
            and the size of the k-th partition subset.
    """

    for partition_subset in partition:
        partition_subset.sort()

    index_set = sorted(sum(partition, start=[]))
    n = index_set[-1]

    assert index_set == list(range(n + 1)), "`partition` is not a proper partition."

    for partition_subset, constraint in zip(partition, constraints, strict=True):
        assert (
            0 < constraint < len(partition_subset)
        ), "There is a trivial or invalid constraint."

    is_compatible = True

    for index, partition_subset in enumerate(partition):

        partition_subset_preimage = sorted(
            permutation.inverse_permutation_array[partition_subset]
        )

        if partition_subset_preimage in partition:
            image_index = partition.index(partition_subset_preimage)
            if constraints[image_index] != constraints[index]:
                is_compatible = False
        else:
            is_compatible = False

    return is_compatible


@register_pytree_node_class
class PermutationOperatorFermion(PermutationOperatorBase):
    """
    Operators corresponding to the permutation of the orbitals belonging to a
    second quantized fermionic space. Used to express lattice symmetries of
    fermionic Hamiltonians and symmetrize variational states with respect to
    these symmetries.

    This permutation operator differs from
    :class:`netket.operator.permutation.PermutationOperator` due to the
    fermionic sign that is required to properly define permutations of fermionic
    orbitals.

    .. warning::

        Only works for spaces with a fixed number of fermions.

    For mathematical details on the definition of a permutation operator
    and its justification, we refer to :doc:`/advanced/symmetry`.
    """

    def __init__(
        self,
        hilbert: SpinOrbitalFermions,
        permutation: Permutation,
        dtype: DType | None = float,
    ):
        """
        Construct the permutation operator.

        Args:
            hilbert: The fermionic Hilbert space.
            permutation: The permutation that defines the operator as
                specified in :doc:`/advanced/symmetry`.
        """
        assert isinstance(hilbert, SpinOrbitalFermions)

        if hilbert.n_fermions is None:
            raise TypeError("The Hilbert space must have a fixed number of fermions.")

        if hilbert.n_fermions_per_spin[0] is not None:

            if (
                sum(
                    constraint == 0 or constraint == hilbert.n_orbitals
                    for constraint in hilbert.n_fermions_per_spin
                )
                > 0
            ):
                warnings.warn(
                    "The Hilbert space contains a spin sector that is "
                    "either empty or fully occupied. The compatibility of the "
                    "permutation could not be checked. The trivial spin sector "
                    "should be removed from the Hilbert space."
                )

            else:
                partition = list(
                    list(range(k * hilbert.n_orbitals, (k + 1) * hilbert.n_orbitals))
                    for k in range(hilbert.n_spin_subsectors)
                )
                is_compatible = check_permutation_compatibility(
                    permutation, partition, list(hilbert.n_fermions_per_spin)
                )
                assert (
                    is_compatible
                ), "The permutation is not compatible with the Hilbert space constraints"

        super().__init__(hilbert, permutation, dtype=dtype)

    def _get_signs(self, x):
        return get_antisymmetric_signs(
            x, self.permutation.inverse_permutation_array, self.hilbert.n_fermions
        )

    def get_conn_padded(self, x):
        r"""Finds the connected elements of the Operator.

        Starting from a batch of quantum numbers :math:`x={x_1, ... x_n}` of
        size :math:`B \times M` where :math:`B` size of the batch and :math:`M`
        size of the hilbert space, finds all states :math:`y_i^1, ..., y_i^K`
        connected to every :math:`x_i`.

        Returns a matrix of size :math:`B \times K_{max} \times M` where
        :math:`K_{max}` is the maximum number of connections for every
        :math:`y_i`.

        .. warning::

            Unlike most other operators defined in NetKet, a permutation operator
            is not Hermitian, and we thus have to be careful about the definition of
            connected elements. NetKet defines connected elements of :math:`x` as the
            configurations :math:`x'` such that :math:`\langle x | P_\sigma | x' \rangle` .
            Therefore, the connected elements are the configurations found in the
            image of :math:`x` by :math:`P_{\sigma^{-1}}` , and not :math:`P_\sigma` .

        Args:
            x : A N-tensor of shape :math:`(...,hilbert.size)` containing
                the batch/batches of quantum numbers :math:`x`.

        Returns:
            **(x_primes, mels)**: The connected states x', in a N+1-tensor and an
            N-tensor containing the matrix elements :math:`O(x,x')`
            associated to each x' for every batch.
        """

        x = jnp.asarray(x)
        connected_elements = x.at[..., None, self.permutation.permutation_array].get(
            unique_indices=True, mode="promise_in_bounds"
        )
        signs = self._get_signs(x).astype(self.dtype)
        return connected_elements, signs[..., jnp.newaxis]

    def trace(self) -> int:
        partition_labels = sum(
            [
                self.hilbert.n_orbitals * [k]
                for k in range(self.hilbert.n_spin_subsectors)
            ],
            start=[],
        )
        cycle_decomposition = self.permutation.cycle_decomposition()
        cycle_occupation = get_subset_occupations(partition_labels, cycle_decomposition)
        return get_parity_sum(cycle_occupation, self.hilbert.n_fermions_per_spin)
