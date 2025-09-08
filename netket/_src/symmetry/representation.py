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
from functools import cached_property

import numpy as np

from netket.utils.group import Element, FiniteGroup
from netket.operator import DiscreteJaxOperator, SumOperator
from netket.vqs.mc.mc_state.state import MCState


class Representation:
    """
    A representation of a group. For a given group, it is effectively a dictionary
    that maps the elements of that group to their corresponding operators.
    """

    def __init__(
        self,
        group: FiniteGroup,
        representation_dict: dict[Element, DiscreteJaxOperator],
    ):
        """
        Constructs a Representation of a group from a dictionary associating
        an operator to every group element.
        """

        if not isinstance(group, FiniteGroup):
            raise TypeError("group must be a FiniteGroup")

        operator = next(iter(representation_dict.values()))
        hilbert = operator.hilbert

        for element, operator in representation_dict.items():
            assert hilbert == operator.hilbert
            assert element in group.elems

        representations = tuple(representation_dict[el] for el in group.elems)

        if not len(group.elems) == len(representation_dict):
            raise ValueError(
                "The representation dictionary must have the same "
                "number of elements as the group."
            )

        self.hilbert = hilbert
        self.group = group
        self.representations = representations

    def __repr__(self):
        return f"Representation(group={self.group}, hilbert={self.hilbert})"

    def __getitem__(self, key):
        if isinstance(key, Element):
            return self.representation_dict[key]
        elif isinstance(key, int):
            return self.representation_dict[self.group[key]]
        raise TypeError("Index should be integer or group element")

    def __hash__(self):
        return hash(("Representation", self.hilbert, self.group, self.representations))

    def __eq__(self, other):
        if type(self) is type(other):
            return (
                self.hilbert == other.hilbert_space
                and self.group == other.group
                and self.representations == other.representations
            )
        return False

    @cached_property
    def representation_dict(self) -> dict[Element, DiscreteJaxOperator]:
        """
        Dictionary associating every group element to a representation

        Equivalent to `{el: rep for (el, rep) in (self.group.elems,
        self.representations)}`
        """
        return {el: rep for (el, rep) in zip(self.group.elems, self.representations)}

    def __iter__(self):
        return iter(self.representation_dict.items())

    def projector(self, character_index: int) -> DiscreteJaxOperator:
        """Build the projection operator corresponding to a given
        irreducible representation."""
        character_table = self.group.character_table()
        prefactor = character_table[character_index, 0] / len(self.group.elems)

        # Build manually the SumOperator for efficiency when operating with
        # large groups
        operators = tuple(self[g] for g in self.group)
        coefficients = prefactor * np.conj(character_table[character_index])
        projector = SumOperator(*operators, coefficients=coefficients)
        return projector

    def project(self, state, character_index: int) -> MCState:
        """Return the state projected onto the subspace associated to the
        irreducible representation specified by character_index."""
        from netket._src.vqs.transformed_vstate import apply_operator

        projector = self.projector(character_index)
        projected_state = apply_operator(projector, state)
        return projected_state

    @property
    def character(self) -> np.ndarray:
        """
        The vector storing the character of the representation.

        Corresponds to ``[op.trace() for (_, op) in self]``.

        Requires that each operator of the representation
        implements the trace method.
        """
        try:
            character = [op.trace() for perm, op in self]
        except NotImplementedError as err:
            raise NotImplementedError(
                "At least one operator of the representation "
                "does not implement the trace. The original error "
                "follows:"
            ) from err
        return np.array(character)

    def irrep_subspace_dims(self) -> int:
        """
        Return the dimension of the subspace associated to each irreducible
        representation.

        Requires that each operator of the representation implements
        the trace method.
        """
        character_table = self.group.character_table()
        group_order = len(self.group.elems)

        irrep_count = character_table @ self.character / group_order
        irrep_dims = np.round(irrep_count * character_table[:, 0]).astype(int)
        return irrep_dims

    def symmetry_adapted_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a tuple `(mat, irrep_dims)`, where `mat` is the change of basis
        matrix associated to a symmetry adapted basis, and `irrep_dims` is
        an array giving the dimension of the subspace associated to each
        irreducible representation.

        The basis vectors of `mat` are ordered by the index of their irreducible
        representation.
        """
        n_irreps = self.group.character_table().shape[0]
        projectors = [self.projector(k).to_dense() for k in range(n_irreps)]

        cob_matrix = np.zeros(
            (self.hilbert.n_states, self.hilbert.n_states), dtype=complex
        )
        current_index = 0

        irrep_dims = []

        for projector in projectors:

            eigvals, eigvecs = np.linalg.eig(projector)
            is_in_image = np.abs(eigvals - 1) < 1e-12
            selected_eigvecs = eigvecs[:, is_in_image]

            projector_rank = selected_eigvecs.shape[1]
            irrep_dims.append(projector_rank)

            cob_matrix[:, current_index : current_index + projector_rank] = (
                selected_eigvecs
            )
            current_index += projector_rank

        return cob_matrix, np.array(irrep_dims)

    # def construct_commuting_product(self, other, check_commute: bool = False):
    #     """
    #     Construct the product representation in the case where all operators of one representation
    #     commute with all operators of the other.
    #     It is a representation of the direct product of the underlying groups.
    #     """
    #     if check_commute:
    #         assert self.is_commuting(other)

    #     # group_direct_product = self.group @ other.group
    #     # operator_products = {G1*G2: self.representation_mapping[g1] * other.representation_mapping[g2] for
    #     #         g1 in self.group for g2 in other.group}

    #     # return Representation(group_direct_product, self.hilbert, operator_products)

    # def is_commuting(self, other):
    #     """
    #     Check whether all operators of one representation commute with all operators of
    #     the other.
    #     """
    #     for operator_1 in self.representation_dict.values():
    #         for operator_2 in other.representation_dict.values():
    #             assert operator_1 @ operator_2 - operator_2 @ operator_1 == 0

    # # We might want a fast mode where only the fast checks are made,
    # # and an exhaustive mode where all checks are made.
    # def check_representation(self):
    #     """
    #     Check whether the representation is valid by checking that the
    #     representation properties are satisfied.
    #     """

    #     is_representation = True

    #     # Identity property
    #     for g in self.group:
    #         if isinstance(g, Identity):
    #             if (
    #                 not jnp.linalg.norm(
    #                     self[g].to_dense - jnp.eye(self.hilbert.n_states)
    #                 )
    #                 < 1e-14
    #             ):
    #                 is_representation = False

    #     # Compatibility property
    #     for g_1, g_2 in product(self.group, self.group):
    #         product_inside = self[g_1 @ g_2]
    #         product_outside = self[g_1] @ self[g_2]
    #         if not product_inside == product_outside:
    #             is_representation = False

    #     return is_representation
