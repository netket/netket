import itertools
import math
import numpy as np

import jax
import jax.numpy as jnp

from netket.utils import struct, StaticRange
from netket.utils.types import Scalar, Array

from ..base import HilbertIndex, is_indexable
from ..unconstrained import LookupTableHilbertIndex
from ..uniform_tensor import UniformTensorProductHilbertIndex

from .base import optimalConstrainedHilbertindex, ConstrainedHilbertIndex


@struct.dataclass
class SumConstraint:
    """
    Constraint of an Hilbert space enforcing a total sum of all the values in the degrees of freedom.

    Constructed by specifying the total sum. For Fock-like spaces this is the total population,
    while for Spin-like spaces this is the magnetisation.
    """

    sum_value: Scalar = struct.field(pytree_node=False)

    @jax.jit
    def __call__(self, x: Array) -> Array:
        return x.sum(axis=1) == self.sum_value


@optimalConstrainedHilbertindex.dispatch
def optimalConstrainedHilbertindex(local_states, size, constraint: SumConstraint):
    # If we have a constraint, we tentatively construct a specialised Hilbert index for that particular constraint.
    # If this specialised indexer object exists, we check whether it is more efficient than the generic
    # ConstrainedHilbertIndex one. If it is more efficient, we use it, otherwise we keep the generic one.
    generic_index = ConstrainedHilbertIndex(
        UniformTensorProductHilbertIndex(local_states, size), constraint
    )
    specialized_index = SumConstrainedHilbertIndex(
        local_states, size, constraint.sum_value
    )
    if specialized_index.n_states_bound < generic_index.n_states_bound:
        return specialized_index
    else:
        return generic_index


@struct.dataclass
class SumConstrainedHilbertIndex(HilbertIndex):
    """
    Specialized implementation for a constrained space with a SumConstraint.
    Does not require the unconstrained space to be indexable.
    """

    range: StaticRange = struct.field(pytree_node=True)
    size: int = struct.field(pytree_node=False)
    sum_value: int = struct.field(pytree_node=False)

    @property
    def shape(self):
        return (self.range.length,) * self.size

    @property
    def n_particles(self):
        # sum_value: e.g. total_sz
        # convert the constraint to a constraint for the number of particles
        # of a fock space with staes in the range [0,1,...,n]
        return round((self.sum_value - self.range.start * self.size) / self.range.step)

    @property
    def n_states(self):
        if self._n_max == 1:
            return math.comb(self.size, self.n_particles)
        else:
            return self._lookup_table.n_states

    @property
    def _n_max(self):
        return max(self.shape) - 1

    def states_to_numbers(self, states: Array) -> Array:
        return self._lookup_table.states_to_numbers(states)

    def numbers_to_states(self, numbers: Array):
        return self._lookup_table.numbers_to_states(numbers)

    def all_states(self):
        return self._lookup_table.all_states()

    @jax.jit
    def _compute_all_states(self):
        if self.n_particles == 0:
            return jnp.zeros((1, self.size), dtype=jnp.int32)
        with jax.ensure_compile_time_eval():
            c = jnp.repeat(
                jnp.eye(self.size, dtype=jnp.int32),
                np.array(self.shape) - 1,
                axis=0,
            )
            combs = jnp.array(
                list(itertools.combinations(np.arange(len(c)), self.n_particles))
            )
            all_states = c[combs].sum(axis=1, dtype=jnp.int32)
            if (np.array(self.shape) > 1).any():
                all_states = jnp.unique(all_states, axis=0)
        all_states_fock = jnp.asarray(all_states)
        return self.range.numbers_to_states(all_states_fock, dtype=np.int32)

    @struct.property_cached(pytree_node=True)
    def _lookup_table(self) -> LookupTableHilbertIndex:
        with jax.ensure_compile_time_eval():
            all_states = self._compute_all_states()
        return LookupTableHilbertIndex(all_states)

    @property
    def n_states_bound(self):
        # upper bound on n_states, exact if n_max == 1
        # number of combinations to check in _compute_all_states
        return math.comb((np.array(self.shape) - 1).sum(), self.n_particles)

    @property
    def is_indexable(self):
        # make sure we have less than than max_states to check in _compute_all_states
        return is_indexable(self.n_states_bound)
