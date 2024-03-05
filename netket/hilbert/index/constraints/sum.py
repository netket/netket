import itertools
import math
import numpy as np

import jax
import jax.numpy as jnp

from netket.utils import struct, StaticRange
from netket.utils.types import Scalar, Array

from ..base import HilbertIndex, is_indexable
from ..unconstrained import LookupTableHilbertIndex
from .registry import register_constrained_hilbert_index


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


class SumConstrainedHilbertIndexFock(HilbertIndex):
    # Fock, supports non-uniform shape
    shape: tuple[int] = struct.field(pytree_node=False)
    n_particles: int = struct.field(pytree_node=False)

    def __init__(self, shape, n_particles):
        self.shape = shape
        self.n_particles = n_particles

    @property
    def n_states(self):
        if self._n_max == 1:
            return math.comb(self.size, self.n_particles)
        else:
            return self._lookup_table.n_states

    @property
    def _n_max(self):
        return max(self.shape) - 1

    @property
    def size(self):
        return len(self.shape)

    def states_to_numbers(self, states: Array) -> Array:
        return self._lookup_table.states_to_numbers(states)

    def numbers_to_states(self, numbers: Array):
        return self._lookup_table.numbers_to_states(numbers)

    def all_states(self):
        return self._lookup_table.all_states()

    def _compute_all_states(self):
        if self.n_particles == 0:
            return jnp.zeros((1, self.size), dtype=jnp.int32)
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
            with jax.ensure_compile_time_eval():
                all_states = jnp.unique(all_states, axis=0)
        return jnp.asarray(all_states)

    @struct.property_cached(pytree_node=True)
    def _lookup_table(self) -> LookupTableHilbertIndex:
        return LookupTableHilbertIndex(self._compute_all_states())

    @property
    def n_states_bound(self):
        # upper bound on n_states, exact if n_max == 1
        # number of combinations to check in _compute_all_states
        return math.comb((np.array(self.shape) - 1).sum(), self.n_particles)

    @property
    def is_indexable(self):
        # make sure we have less than than max_states to check in _compute_all_states
        return is_indexable(self.n_states_bound)


class SumConstrainedHilbertIndex(SumConstrainedHilbertIndexFock):
    # shape is uniform, with same StaticRange on all sites
    _range: StaticRange

    def __init__(self, range_, size, sum_value):
        # sum_value: e.g. total_sz
        self.shape = (range_.length,) * size
        # convert the constraint to a constraint on the range [0,1,...,n]
        self.n_particles = round((sum_value - range_.start * size) / range_.step)

        self._range = range_

    def _compute_all_states(self):
        all_states_fock = super()._compute_all_states()
        return self._range.numbers_to_states(all_states_fock, dtype=np.int32)


register_constrained_hilbert_index(
    SumConstraint,
    lambda constraint, local_states, size: SumConstrainedHilbertIndex(
        local_states, size, constraint.sum_value
    ),
)
