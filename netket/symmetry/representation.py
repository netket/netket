import jax.numpy as jnp

from netket.utils import struct

from netket.utils.types import Array
from netket.utils.group import FiniteGroup, Element
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
    hilbert: AbstractHilbert
    representation_mapping: dict[Element, DiscreteJaxOperator]

    def __pre_init__(self, *args, **kwargs):
        # Check that group and representation_mapping.keys() match
        return (args, kwargs)

    """
    @property hilbert_space
    @property group
    @property representation_mapping
    """

    def get_representation_element(self, g: Element) -> DiscreteJaxOperator:
        """
        Return the representation element of g.
        """
        return self.representation_mapping[g]

    def get_character_table(self) -> Array:
        return self.group.get_character_table()

    def get_projector(self, character_index) -> DiscreteJaxOperator:
        """
        Build the projector operator corresponding to a given irreducible representation.
        """
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
        pass
