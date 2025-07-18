import jax.numpy as jnp

from jax.tree_util import register_pytree_node_class

from netket.operator import DiscreteJaxOperator
from netket.utils.group import Permutation


@register_pytree_node_class
class PermutationOperator(DiscreteJaxOperator):
    """Be careful about the permutation being possibly inverted."""

    def __init__(self, hilbert, permutation):
        super().__init__(hilbert)
        assert isinstance(
            permutation, Permutation
        ), "permutation must be a Permutation object."
        self.permutation = permutation

    def tree_flatten(self):
        name = self.permutation._name
        inverse_permutation_array = self.permutation.inverse_permutation_array
        struct_data = {"hilbert": self.hilbert, "name": name}
        return (inverse_permutation_array,), struct_data

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        permutation = Permutation(
            inverse_permutation_array=array_data[0], name=struct_data["name"]
        )
        return cls(struct_data["hilbert"], permutation)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def dtype(self):
        return int

    def __repr__(self):
        if self.permutation._name is not None:
            return f"PermutationOperator({self.permutation._name}: {self.permutation.permutation_array})"
        else:
            return f"PermutationOperator({self.permutation.permutation_array})"

    def get_conn_padded(self, x):
        batch_shape, phys_dim = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, phys_dim)
        connected_elements = x.T[self.permutation.inverse_permutation_array].T
        connected_elements = connected_elements.reshape((*batch_shape, 1, phys_dim))
        return connected_elements, jnp.ones((*batch_shape, 1))
