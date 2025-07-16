import jax.numpy as jnp

from netket.utils import struct

from netket.utils.types import Array
from netket.utils.group import Element
from netket.utils.group import FiniteGroup
from netket.hilbert import AbstractHilbert
from netket.operator import DiscreteJaxOperator


@struct.dataclass
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

    def __pre_init__(
        self,
        group: FiniteGroup,
        representation_dict: dict[Element, DiscreteJaxOperator],
    ):

        operator = next(iter(representation_dict.values()))
        hilbert_space = operator.hilbert

        for element, operator in representation_dict.items():
            assert hilbert_space == operator.hilbert
            assert element in group.elems

        return (
            (),
            dict(
                group=group,
                hilbert_space=hilbert_space,
                representation_dict=representation_dict,
            ),
        )

    """
    @property hilbert_space
    @property group
    @property representation_dict
    """

    def get_representation_element(self, g: Element) -> DiscreteJaxOperator:
        """Return the operator corresponding to the group element g."""
        return self.representation_dict[g]

    def get_character_table(self) -> Array:
        """Return the character table of the group of this representation."""
        return self.group.character_table_readable()

    def get_projector(self, character_index) -> DiscreteJaxOperator:
        """Build the projector operator corresponding to a given irreducible representation."""
        character_table = self.get_character_table()
        return sum(
            [
                jnp.conj(character_table[character_index, element_index])
                * self.representation_mapping[g]
                for element_index, g in enumerate(self.group)
            ]
        )

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

    def check_representation(self):
        """
        Check whether the representation is valid by checking that the
        multiplication table of the operators matches that of the group.
        """
        pass
