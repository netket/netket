# Copyright 2024 The NetKet Authors - All rights reserved.
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


import jax
import jax.numpy as jnp


from netket.utils.types import Array
from netket.utils import HashableArray, struct
from netket.jax import sort, searchsorted

from .base import HilbertIndex, is_indexable


class LookupTableHilbertIndex(HilbertIndex):
    """Index states according to a pre-defined array containing all possible states.

    Does lookup (states_to_numbers) in log time in the number of states,
    and indexing (numbers_to_states) in constant time.

    The pre-defined array is sorted internally in ascending order (lexicographically).
    Lookup of states not in all_states results in undefined behaviour.
    """

    _all_states: HashableArray = struct.field(pytree_node=False)

    def __init__(self, all_states: Array):
        self._all_states = HashableArray(all_states)

    @property
    def n_states(self) -> int:
        return self.all_states().shape[0]

    @jax.jit
    def numbers_to_states(self, numbers: Array) -> Array:
        return self.all_states()[numbers]

    @jax.jit
    def states_to_numbers(self, states: Array) -> Array:
        return searchsorted(self.all_states(), states)

    @jax.jit
    def all_states(self) -> Array:
        with jax.ensure_compile_time_eval():
            # unpack HashableArray,
            # ensure the states are sorted
            res = sort(jnp.asarray(self._all_states))
        return res

    @property
    def is_indexable(self) -> bool:
        return is_indexable(self.n_states)
