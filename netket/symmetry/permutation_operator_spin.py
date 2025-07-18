import jax.numpy as jnp

from jax.tree_util import register_pytree_node_class

from netket.operator import DiscreteJaxOperator


@register_pytree_node_class
class PermutationOperator(DiscreteJaxOperator):
    """Be careful about the permutation being possibly inverted."""

    def __init__(self, hilbert, permutation):
        super().__init__(hilbert)
        self.permutation = permutation

    def tree_flatten(self):
        array_data = self.permutation
        struct_data = {"hilbert": self.hilbert}
        return array_data, struct_data

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        ...
        return cls(array_data["hilbert"], ...)

    @property
    def max_conn_size(self) -> int:
        return 1

    def dtype(self):
        return int

    def get_conn_padded(self, x):
        batch_shape, phys_dim = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, phys_dim)
        connected_elements = x.T[self.permutation].T
        connected_elements = connected_elements.reshape((*batch_shape, 1, phys_dim))
        return connected_elements, jnp.ones((*batch_shape, 1))
