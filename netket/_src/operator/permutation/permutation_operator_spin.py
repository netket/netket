import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.hilbert import Qubit, Spin
from netket.hilbert.constraint import SumConstraint

from .permutation_operator_base import PermutationOperatorBase
from .trace_utils import count_n_uplets


@register_pytree_node_class
class PermutationOperator(PermutationOperatorBase):
    """
    Permutation operator on a spin or boson space.
    For mathematical details on the definition of a permutation operator
    and its justification, we refer to [SOMEWHERE].

    Args:
        hilbert: The Hilbert space.
        permutation: The permutation represented by the operator.
    """

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
        x = jnp.asarray(x)
        # Check that the parameters of get are useful
        connected_elements = x.at[..., None, self.permutation.permutation_array].get(
            unique_indices=True, mode="promise_in_bounds"
        )
        return connected_elements, jnp.ones((*x.shape[:-1], 1), dtype=self.dtype)

    def __matmul__(self, other):
        if isinstance(other, PermutationOperator):
            return PermutationOperator(
                self.hilbert, self.permutation @ other.permutation
            )
        else:
            return super().__matmul__(other)

    def trace(self):
        cycle_decomposition = self.permutation.get_cycle_decomposition()

        if isinstance(self.hilbert, Qubit):
            return 2 ** len(cycle_decomposition)

        if isinstance(self.hilbert, Spin):

            if not self.hilbert.constrained:
                return self.hilbert.local_size ** len(cycle_decomposition)

            elif isinstance(self.hilbert.constraint, SumConstraint):
                cycle_lengths = [len(cycle) for cycle in cycle_decomposition]
                shifted_constraint = (
                    self.hilbert.constraint.sum_value
                    + self.hilbert.size * (self.hilbert.local_size - 1)
                )
                return count_n_uplets(
                    cycle_lengths, shifted_constraint, self.hilbert.local_size
                )

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
