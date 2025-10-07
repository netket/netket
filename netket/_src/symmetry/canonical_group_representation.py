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

from typing import TYPE_CHECKING

from netket.hilbert import DiscreteHilbert
from netket.utils.group import PermutationGroup

if TYPE_CHECKING:
    from netket.symmetry import Representation


def canonical_group_representation(
    hilbert: DiscreteHilbert, group: PermutationGroup
) -> "Representation":
    """
    Construct the canonical representation of a permutation group on a many-body
    Hilbert space.

    This function creates a representation of a permutation group that acts on the
    physical sites of a lattice or graph, extended to act on the full many-body
    Hilbert space. The representation operators depend on the Hilbert space type.

    For spin Hilbert spaces, permutations act directly on the spins. For fermionic
    Hilbert spaces, the permutations are extended to match the number of single-particle
    states, and appropriate fermionic signs are included.

    .. note::
        This function is **experimental** and its API may change in future releases.

    .. note::
        This function constructs the canonical representation of lattice symmetry groups
        as permutation groups acting on the lattice sites. For other types of symmetry
        groups (e.g., continuous symmetries, non-spatial symmetries), this representation
        is not defined and you should construct the appropriate representation manually.

    Args:
        hilbert: The Hilbert space on which the representation acts. This can be any
            discrete Hilbert space, including spin systems (e.g., :class:`~netket.hilbert.Spin`,
            :class:`~netket.hilbert.Qubit`) or fermionic systems
            (:class:`~netket.hilbert.SpinOrbitalFermions`).
        group: A :class:`~netket.utils.group.PermutationGroup` object describing the
            symmetry group. This can be obtained from lattice methods like
            :meth:`~netket.graph.Lattice.translation_group`,
            :meth:`~netket.graph.Lattice.point_group`,
            :meth:`~netket.graph.Lattice.space_group`, etc., or be a custom permutation group.

    Returns:
        A :class:`~netket.symmetry.Representation` object encoding the action of the
        permutation group on the Hilbert space. The representation includes a dictionary
        mapping each group element to the corresponding operator that acts on quantum states.

    Examples:
        Get the representation of the translation group on a spin system:

        >>> import netket as nk
        >>> lattice = nk.graph.Square(4)
        >>> hilbert = nk.hilbert.Spin(0.5, N=lattice.n_nodes)
        >>> trans_group = lattice.translation_group()
        >>> rep = nk.experimental.symmetry.canonical_group_representation(hilbert, trans_group)

        Get the representation of a partial symmetry (translations along x only):

        >>> trans_x = lattice.translation_group(dim=0)
        >>> rep_x = nk.experimental.symmetry.canonical_group_representation(hilbert, trans_x)

        Use a point group symmetry (e.g., C4 rotations):

        >>> pg = lattice.point_group(nk.symmetry.group.planar.C(4))
        >>> rep_c4 = nk.experimental.symmetry.canonical_group_representation(hilbert, pg)

        Use the full space group (translations + point group):

        >>> sg = lattice.space_group()
        >>> rep_sg = nk.experimental.symmetry.canonical_group_representation(hilbert, sg)

        This approach works with any :class:`~netket.utils.group.PermutationGroup`,
        allowing you to construct representations for specific subgroups of the full
        symmetry group or even use custom permutation groups.

        For fermionic systems, the permutations are automatically extended to account
        for multiple spin species:

        >>> fermion_hilbert = nk.hilbert.SpinOrbitalFermions(
        ...     n_orbitals=lattice.n_nodes,
        ...     s=1/2,
        ...     n_fermions_per_spin=(4, 4)
        ... )
        >>> rep_fermion = nk.experimental.symmetry.canonical_group_representation(
        ...     fermion_hilbert, trans_group
        ... )

    See Also:
        - :class:`~netket.symmetry.Representation`: The representation class that is returned.
        - :class:`~netket.utils.group.PermutationGroup`: Permutation group class.
        - :meth:`~netket.graph.Lattice.translation_group`: Get translation group of a lattice.
        - :meth:`~netket.graph.Lattice.point_group`: Get point group of a lattice.
        - :meth:`~netket.graph.Lattice.space_group`: Get space group of a lattice.
    """
    from netket._src.symmetry.representation_construction import (
        physical_to_many_body_permutation_group,
        permutation_group_representation,
    )

    # If fermionic hilbert space, increase the size of the permutation so it
    # matches the number of single-particle states.
    group = physical_to_many_body_permutation_group(group, hilbert)
    return permutation_group_representation(hilbert, group)
