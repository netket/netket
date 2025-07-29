import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.hilbert import AbstractHilbert
from netket.operator import DiscreteJaxOperator
from netket.utils.group import Permutation, Identity


@register_pytree_node_class
class PermutationOperatorBase(DiscreteJaxOperator):
    """
    Abstract permutation operator for either spin of fermion space.
    For mathematical details on the definition of a permutation operator
    and its justification, we refer to [SOMEWHERE].

    Args:
        hilbert: The Hilbert space.
        permutation: The permutation represented by the operator.
    """

    def __init__(self, hilbert: AbstractHilbert, permutation: Permutation):
        super().__init__(hilbert)
        if isinstance(permutation, Identity):
            permutation = Permutation(
                permutation_array=jnp.arange(hilbert.size), name="Identity"
            )

        if not isinstance(permutation, Permutation):
            raise TypeError("permutation must be a Permutation object.")
        if not hilbert.size == permutation.permutation_array.size:
            raise ValueError(
                "Permutation size does not correspond to Hilbert space size."
            )

        self.permutation = permutation

    def tree_flatten(self):
        struct_data = {"hilbert": self.hilbert, "permutation": self.permutation}
        return (), struct_data

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        return cls(**struct_data)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def dtype(self):
        return jnp.float32

    def __hash__(self):
        return hash((self.hilbert, self.permutation))
