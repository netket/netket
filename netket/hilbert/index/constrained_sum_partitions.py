import math

import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import Array

from netket.hilbert.constraint import SumOnPartitionConstraint

from .base import HilbertIndex, is_indexable
from .unconstrained import LookupTableHilbertIndex
from .constrained_sum import SumConstrainedHilbertIndex
from .constrained_generic import optimalConstrainedHilbertindex


@optimalConstrainedHilbertindex.dispatch
def optimalConstrainedHilbertindex(
    local_states, size, constraint: SumOnPartitionConstraint
):
    if size != sum(constraint.sizes):
        raise ValueError(
            f"The size {size} does not match the total size of the constraint"
            f"partitions {sum(constraint.sizes)}."
        )
    specialized_index = SumOnPartitionConstrainedHilbertIndex(
        sub_indices=tuple(
            SumConstrainedHilbertIndex(local_states, N, sv)
            for (N, sv) in zip(constraint.sizes, constraint.sum_values)
        )
    )
    return specialized_index


@struct.dataclass
class SumOnPartitionConstrainedHilbertIndex(HilbertIndex):
    """
    Specialized implementation for a constrained space with a SumConstraint.
    Does not require the unconstrained space to be indexable.
    """

    sub_indices: list[SumConstrainedHilbertIndex] = struct.field(pytree_node=False)

    @property
    def size(self) -> int:
        return sum(s.size for s in self.sub_indices)

    @property
    def shape(self):
        return sum((s.shape for s in self.sub_indices), start=())

    @property
    def n_states(self):
        return math.prod(s.n_states for s in self.sub_indices)

    def states_to_numbers(self, states: Array) -> Array:
        return self._lookup_table.states_to_numbers(states)

    def numbers_to_states(self, numbers: Array):
        return self._lookup_table.numbers_to_states(numbers)

    def all_states(self):
        return self._lookup_table.all_states()

    @jax.jit
    def _compute_all_states(self):
        states = []
        for index in self.sub_indices:
            states.append(index.all_states())
        return combine_configurations(*states)

    @struct.property_cached(pytree_node=True)
    def _lookup_table(self) -> LookupTableHilbertIndex:
        with jax.ensure_compile_time_eval():
            all_states = self._compute_all_states()
        return LookupTableHilbertIndex(all_states)

    @property
    def n_states_bound(self):
        # upper bound on n_states, exact if n_max == 1
        # number of combinations to check in _compute_all_states
        return math.prod(s.n_states_bound for s in self.sub_indices)

    @property
    def is_indexable(self):
        # make sure we have less than than max_states to check in _compute_all_states
        return is_indexable(self.n_states_bound)


def combine_configurations(*matrices):
    """
    Concatenate a list of configurations, generating all possible combinations.
    """
    if len(matrices) == 1:
        return matrices[0]

    # Initialize with the first matrix
    result = matrices[0]

    for matrix in matrices[1:]:
        M1, N1 = result.shape
        M2, N2 = matrix.shape

        # Reshape matrices to make use of broadcasting
        result_reshaped = result[:, jnp.newaxis, :]
        matrix_reshaped = matrix[jnp.newaxis, :, :]

        # Create the resulting matrix by concatenating along the last axis
        result = jnp.concatenate(
            (result_reshaped.repeat(M2, axis=1), matrix_reshaped.repeat(M1, axis=0)),
            axis=-1,
        )

        # Reshape to the new form
        result = result.reshape(M1 * M2, N1 + N2)

    return result
