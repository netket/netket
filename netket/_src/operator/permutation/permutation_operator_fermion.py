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

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from functools import partial

from netket.hilbert import SpinOrbitalFermions
from netket.utils.group import Permutation

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


@register_pytree_node_class
class PermutationOperatorFermion(PermutationOperatorBase):
    """
    Permutation operator on a fermion space.
    ONLY WORKS FOR A HILBERT SPACE WITH FIXED NUMBER OF FERMIONS.
    For mathematical details on the definition of a permutation operator
    and its justification, we refer to :doc:`/advanced/symmetry`.

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

    def _get_signs(self, x):
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
        signs = self._get_signs(x).astype(jnp.float32)
        return connected_elements, signs[..., jnp.newaxis]

    def trace(self) -> float:
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
