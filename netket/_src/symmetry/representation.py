import numpy as np
import jax.numpy as jnp

from functools import reduce
from itertools import product


from netket.utils.group import Element, FiniteGroup, Identity
from netket.hilbert import AbstractHilbert
from netket.operator import DiscreteJaxOperator


class Representation:
    """
    A representation of a group
        - group
        - hilbert space
        - a dictionary that maps the elements of the given group to their corresponding operators.

        The dictionary should hold an operator for every group element.
    """

    group: FiniteGroup
    hilbert_space: AbstractHilbert
    representation_dict: dict[Element, DiscreteJaxOperator]

    # Fix the __init__
    def __init__(
        self,
        group: FiniteGroup,
        representation_dict: dict[Element, DiscreteJaxOperator],
    ):

        operator = next(iter(representation_dict.values()))
        hilbert_space = operator.hilbert

        for element, operator in representation_dict.items():
            assert hilbert_space == operator.hilbert
            assert element in group.elems

        assert len(group.elems) == len(representation_dict)

        self.hilbert_space = hilbert_space
        self.group = group
        self.representation_dict = representation_dict

    def __getitem__(self, key):
        if isinstance(key, Element):
            return self.representation_dict[key]
        elif isinstance(key, int):
            return self.representation_dict[self.group[key]]
        raise TypeError("Index should be integer or group element")

    def __iter__(self):
        return iter(self.representation_dict.items())

    def get_projector(self, character_index) -> DiscreteJaxOperator:
        """Build the projection operator corresponding to a given irreducible representation."""
        character_table = self.group.character_table()
        prefactor = character_table[character_index, 0] / len(self.group.elems)
        operator_list = [
            jnp.conj(character_table[character_index, element_index]) * self[g]
            for element_index, g in enumerate(self.group)
        ]
        projector = prefactor * reduce(lambda x, y: x + y, operator_list)
        return projector

    def get_character(self):
        """
        Return the character of the representation. Requires that each operator
        of the representation implements the trace method.
        """
        try:
            character = [op.trace() for perm, op in self]
        except NotImplementedError:
            raise NotImplementedError(
                "At least one operator of the representation"
                " does not implement the trace"
            )
        return np.array(character)

    def get_irrep_subspace_dims(self):
        """
        Return the dimension of the subspace associated to each irreducible
        representation. Requires that each operator of the representation
        implements the trace method.
        """
        character = self.get_character()
        character_table = self.group.character_table()
        group_order = len(self.group.elems)

        irrep_count = character_table @ character / group_order
        irrep_dims = np.round(irrep_count * character_table[:, 0]).astype(int)
        return irrep_dims

    def get_symmetry_adapted_basis(self):
        """
        Return a tuple `(mat, irrep_dims)`, where `mat` is the change of basis
        matrix associated to a symmetry adapted basis, and `irrep_dims` is
        an array giving the dimension of the subspace associated to each
        irreducible representation.
        The basis vectors of `mat` are ordered by the index of their irreducible
        representation.
        """
        n_irreps = self.group.character_table().shape[0]
        projectors = [self.get_projector(k).to_dense() for k in range(n_irreps)]

        cob_matrix = np.zeros(
            (self.hilbert_space.n_states, self.hilbert_space.n_states), dtype=complex
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

    def construct_commuting_product(self, other, check_commute: bool = False):
        """
        Construct the product representation in the case where all operators of one representation
        commute with all operators of the other.
        It is a representation of the direct product of the underlying groups.
        """
        if check_commute:
            assert self.is_commuting(other)

        # group_direct_product = self.group @ other.group
        # operator_products = {G1*G2: self.representation_mapping[g1] * other.representation_mapping[g2] for
        #         g1 in self.group for g2 in other.group}

        # return Representation(group_direct_product, self.hilbert_space, operator_products)

    def is_commuting(self, other):
        """
        Check whether all operators of one representation commute with all operators of
        the other.
        """
        for operator_1 in self.representation_dict.values():
            for operator_2 in other.representation_dict.values():
                assert operator_1 @ operator_2 - operator_2 @ operator_1 == 0

    # We might want a fast mode where only the fast checks are made,
    # and an exhaustive mode where all checks are made.
    def check_representation(self):
        """
        Check whether the representation is valid by checking that the
        representation properties are satisfied.
        """

        is_representation = True

        # Identity property
        for g in self.group:
            if isinstance(g, Identity):
                if (
                    not jnp.linalg.norm(
                        self[g].to_dense - jnp.eye(self.hilbert_space.n_states)
                    )
                    < 1e-14
                ):
                    is_representation = False

        # Compatibility property
        for g_1, g_2 in product(self.group, self.group):
            product_inside = self[g_1 @ g_2]
            product_outside = self[g_1] @ self[g_2]
            if not product_inside == product_outside:
                is_representation = False

        return is_representation
