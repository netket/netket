# Copyright 2023-2024 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math
import numpy as np

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from typing import Union
from netket.utils.types import Array
from netket.utils import struct, StaticRange

from .base import HilbertIndex, is_indexable
from .unconstrained import LookupTableHilbertIndex


# This function has exponential runtime in self.size, so we cache it in order to
# only compute it once.
# TODO: distribute over devices/MPI (expensive constraint_fun),  choose better chunk size
@partial(jax.jit, static_argnames=("chunk_size"))
def compute_constrained_to_bare_conversion_table(
    hilbert_index: HilbertIndex,
    constraint_fun: Callable[[Array], Array],
    *,
    chunk_size: int = 65536,
):
    """
    Computes the conversion table that converts the 'constrained' indices
    of an hilbert space to bare indices, so that routines generating
    only values in an unconstrained space can be used.

    Args:
        hilbert_index:
            A dataclass with only metadata (only pytree_node=False)
        constraint_fun:
            A dataclass with only metadata (only pytree_node=False) and __call__ attribute
            Python functions can be used by wrapping them in a jax.tree_util.Partial
            with no args and keywords.
        chunk_size: (optional, default=65536)
            This function operates on blocks of `chunk_size` states at a time in order
            to lower the memory cost. The default chunk size has been chosen arbitrarily
            and might need tweaking depending on the particular constraint_fun.
    """

    with jax.ensure_compile_time_eval():
        n_chunks = int(np.ceil(hilbert_index.n_states / chunk_size))
        bare_number_chunks = []
        for i in range(n_chunks):
            id_start = chunk_size * i
            id_end = np.minimum(chunk_size * (i + 1), hilbert_index.n_states)
            ids = jnp.arange(id_start, id_end, dtype=jnp.int32)
            states = hilbert_index.numbers_to_states(ids)
            is_constrained = constraint_fun(states)
            (chunk_bare_number,) = jnp.nonzero(is_constrained)
            bare_number_chunks.append(chunk_bare_number + id_start)
        bare_numbers = jnp.concatenate(bare_number_chunks)
    return bare_numbers


@struct.dataclass
class ConstrainedHilbertIndex(HilbertIndex):
    unconstrained_index: HilbertIndex
    constraint_fun: Callable[[Array], Array] = struct.field(pytree_node=False)

    @struct.property_cached(pytree_node=True)
    def _bare_numbers(self) -> Array:
        return compute_constrained_to_bare_conversion_table(
            self.unconstrained_index, self.constraint_fun
        )

    @property
    def n_states(self) -> int:
        return self._bare_numbers.shape[0]

    @property
    def size(self) -> int:
        return self.unconstrained_index.size

    @property
    def local_states(self) -> Union[Array, StaticRange]:
        return self.unconstrained_index.local_states

    @property
    def local_size(self) -> int:
        return self.unconstrained_index.local_size

    @jax.jit
    def _states_to_numbers(self, states: Array) -> Array:
        out = self.unconstrained_index.states_to_numbers(states)
        return jnp.searchsorted(self._bare_numbers, out)

    def states_to_numbers(self, states: Array) -> Array:
        return self._states_to_numbers(states)

    @jax.jit
    def _numbers_to_states(self, numbers: Array) -> Array:
        # convert to original space
        numbers = self._bare_numbers[numbers]
        return self.unconstrained_index.numbers_to_states(numbers)

    def numbers_to_states(self, numbers: Array) -> Array:
        return self._numbers_to_states(numbers)

    def all_states(self) -> Array:
        return self.numbers_to_states(jnp.arange(self.n_states))

    @property
    def is_indexable(self) -> bool:
        return self.unconstrained_index.is_indexable

    @property
    def n_states_bound(self) -> int:
        # upper bound on n_states
        return self.unconstrained_index.n_states


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
