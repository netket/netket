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

import numpy as np

from netket.utils.group import PermutationGroup, Identity, Permutation, Element
from netket.hilbert import DiscreteHilbert, SpinOrbitalFermions


def _physical_to_fermionic_permutation(
    perm: Element, hilbert: SpinOrbitalFermions
) -> Element:
    """Converts a permutation of the lattice sites to a permutation of single-particles states
    in a fermionic Hilbert space."""

    assert isinstance(perm, Permutation)

    perm_array = perm.permutation_array
    fermionic_perm_array = np.copy(perm_array)

    offset = 0
    while offset < hilbert.size - hilbert.n_orbitals:
        offset += hilbert.n_orbitals
        fermionic_perm_array = np.concatenate(
            (fermionic_perm_array, perm_array + offset)
        )

    return Permutation(permutation_array=fermionic_perm_array)


def physical_to_logical_permutation_group(
    perm_group: PermutationGroup, hilbert: DiscreteHilbert
) -> PermutationGroup:
    """Converts a permutation group of the lattice sites to a permutation group
    of the local degrees of freedom on the many-body Hilbert space."""

    if isinstance(hilbert, SpinOrbitalFermions):
        id_perm = np.arange(hilbert.size)
        fermionic_perms = []

        for perm in perm_group:
            if isinstance(perm, Identity):
                fermionic_perms.append(Permutation(permutation_array=id_perm))

            else:
                fermionic_perm = _physical_to_fermionic_permutation(perm, hilbert)
                fermionic_perms.append(fermionic_perm)

        perm_group = PermutationGroup(fermionic_perms, degree=hilbert.size)

    return perm_group
