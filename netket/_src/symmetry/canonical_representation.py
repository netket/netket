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

from netket.hilbert import DiscreteHilbert, Qubit, Spin, Fock, SpinOrbitalFermions
from netket.symmetry.group import PermutationGroup, PointGroup
from netket.graph.space_group import TranslationGroup, SpaceGroup

from netket._src.symmetry.representation import Representation
from netket._src.symmetry.representation_construction import (
    physical_to_logical_permutation_group,
)
from netket._src.operator.permutation.construct import (
    construct_permutation_operator,
)


def canonical_representation(
    hilbert: DiscreteHilbert, group: PermutationGroup, warn: bool = True
) -> Representation:
    r"""
    Construct the representation of a permutation group on a many-body
    Hilbert space where each permutation is mapped to the permutation operator
    that permutes the local operators.

    This function creates a representation of a permutation group that acts on the
    physical sites of a lattice or graph, extended to act on the full many-body
    Hilbert space. The representation operators depend on the Hilbert space type.

    More precisely, on a spin Hilbert space, the permutation operator associated
    to the permutation :math:`\sigma` is the operator :math:`P_\sigma` such that
    for any local operator :math:`A_k` acting on site :math:`k`, the permutation
    operator transforms it as :math:`P_\sigma A_k P_\sigma^\dagger = A_{\sigma(k)}`.
    For a fermionic Hilbert space, the operator :math:`P_\sigma` is defined such
    that for any site :math:`k`, the creation operator :math:`c_k^\dagger`
    transforms as :math:`P_\sigma c_k^\dagger P_\sigma^\dagger = c_{\sigma(k)}^\dagger`.
    The relevant permutation operators are therefore different depending on the
    type of Hilbert space. For more details, see :doc:`/advanced/symmetry`.

    .. note::
        This function is **experimental** and its API may change in future releases.

    .. warning::
        This function constructs the specific representation of a permutation
        group described above and in the documentation. For other non-spatial
        symmetries, the appropriate representation can be constructed manually
        using the constructor of :class:`~netket.symmetry.Representation`.
        To safeguard against mistaken uses of this function, passing a
        :class:`~netket.symmetry.group.PermutationGroup` that is not also an
        instance of a subclass designed specifically to describe a subgroup of
        the space group, that is, :class:`~netket.graph.space_group.SpaceGroup`,
        :class:`~netket.symmetry.group.PointGroup`, or
        :class:`~netket.graph.space_group.TranslationGroup`, will raise a warning.

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
        warn: If False, disable the warning on passing a
            :class:`~netket.symmetry.group.PermutationGroup` that is not a
            :class:`~netket.graph.space_group.SpaceGroup`,
            a :class:`~netket.symmetry.group.PointGroup`, or a
            :class:`~netket.graph.space_group.TranslationGroup`.

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
        >>> rep = nk.symmetry.canonical_representation(hilbert, trans_group)

        Get the representation of a partial symmetry (translations along x only):

        >>> trans_x = lattice.translation_group(dim=0)
        >>> rep_x = nk.symmetry.canonical_representation(hilbert, trans_x)

        Use a point group symmetry (e.g., C4 rotations):

        >>> pg = lattice.point_group(nk.symmetry.group.planar.C(4))
        >>> rep_c4 = nk.symmetry.canonical_representation(hilbert, pg)

        Use the full space group (translations + point group):

        >>> sg = lattice.space_group()
        >>> rep_sg = nk.symmetry.canonical_representation(hilbert, sg)

        This approach works with any :class:`~netket.utils.group.PermutationGroup`,
        allowing you to construct representations for specific subgroups of the full
        symmetry group or even use custom permutation groups, though in the latter
        case, care should be taken to only use this function when intending to
        construct the specific representation of that custom permutation group
        detailed above.

        For fermionic systems, the permutations are automatically extended to account
        for multiple spin species:

        >>> fermion_hilbert = nk.hilbert.SpinOrbitalFermions(
        ...     n_orbitals=lattice.n_nodes,
        ...     s=1/2,
        ...     n_fermions_per_spin=(4, 4)
        ... )
        >>> rep_fermion = nk.symmetry.canonical_representation(
        ...     fermion_hilbert, trans_group
        ... )

    See Also:
        - :class:`~netket.symmetry.Representation`: The representation class that is returned.
        - :class:`~netket.utils.group.PermutationGroup`: Permutation group class.
        - :meth:`~netket.graph.Lattice.translation_group`: Get the translation group of a lattice.
        - :meth:`~netket.graph.Lattice.space_group`: Get the space group of a lattice.
        - :meth:`~netket.graph.Lattice.point_group`: Get the point group of a lattice.
    """

    if not isinstance(group, PermutationGroup):
        raise NotImplementedError(
            "Only `PermutationGroup` are supported (for now,\n"
            "please open an issue to report what new groups we should support)."
        )

    if not isinstance(hilbert, (Qubit, Spin, Fock, SpinOrbitalFermions)):
        raise ValueError(
            "The permutation operators of this representation "
            "are only defined for a Hilbert space of the class Qubit, Spin, Fock, "
            "or SpinOrbitalFermions."
        )
    if isinstance(hilbert, (Qubit, Spin, Fock)) and not group.degree == hilbert.size:
        raise ValueError(
            "In the case of a spin Hilbert space, the permutations of group "
            "should be permutations over hilbert.size elements."
        )
    if (
        isinstance(hilbert, (SpinOrbitalFermions))
        and not group.degree == hilbert.n_orbitals
    ):
        raise ValueError(
            "In the case of a fermionic Hilbert space, the permutations of "
            "group should be permutations over hilbert.n_orbitals elements."
        )
    if warn and not isinstance(group, (SpaceGroup, TranslationGroup, PointGroup)):
        warnings.warn(
            "This function constructs a specific representation "
            "of the given permutation group that corresponds to spatial "
            "symmetries, as described above and in the documentation.\n\n"
            "Make sure that this is the intended representation. "
            "To disable this warning, pass a group of the class "
            "SpaceGroup, PointGroup, or TranslationGroup, "
            "or pass no_warning=True.\n\n"
        )

    # If fermionic hilbert space, increase the size of the permutation so it
    # matches the number of single-particle states.
    group = physical_to_logical_permutation_group(group, hilbert)
    return Representation(
        group, {perm: construct_permutation_operator(hilbert, perm) for perm in group}
    )
