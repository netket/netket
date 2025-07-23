import jax.numpy as jnp

from jax.tree_util import register_pytree_node_class

from netket.operator import DiscreteJaxOperator
from netket.utils.group import Permutation, Identity


@register_pytree_node_class
class PermutationOperator(DiscreteJaxOperator):
    """Be careful about the permutation being possibly inverted."""

    def __init__(self, hilbert, permutation):
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

    def __repr__(self):
        if self.permutation._name is not None:
            return f"PermutationOperator({self.permutation._name}: {self.permutation.permutation_array})"
        else:
            return f"PermutationOperator({self.permutation.permutation_array})"

    def __eq__(self, other):
        if isinstance(other, PermutationOperator):
            return (
                self.hilbert == other.hilbert and self.permutation == other.permutation
            )
        else:
            return False

    def get_conn_padded(self, x):
        # Check that the parameters of get are useful
        connected_elements = x.at[..., None, self.permutation.permutation_array].get(
            unique_indices=True, mode="promise_in_bounds"
        )
        return connected_elements, jnp.ones((*x.shape[:-1], 1), dtype=self.dtype)
