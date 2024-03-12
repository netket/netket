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

from netket.utils import struct
from netket.utils.types import Array

from .base import HilbertIndex, is_indexable


@struct.dataclass
class UniformTensorProductHilbertIndex(HilbertIndex):
    """
    Indexes a tensor product of multiple HilbertIndexes, ordered in LSB order
    (so when increasing by one the index, the right-most degree of freedom is increased).
    This index assumes a tensor product of the same Hilbert index with itself _size times,
    and is used to represent all UniformHilbert spaces.
    """

    # tensor product with uniform local space
    local_index: HilbertIndex
    size: int = struct.field(pytree_node=False)

    @property
    def local_size(self) -> int:
        return self.local_index.n_states

    @property
    def n_states(self) -> int:
        return self.local_size**self.size

    @property
    def local_states(self) -> Array:
        return self.local_index.all_states()

    @property
    def _basis(self) -> Array:
        return self.local_size ** jax.lax.iota(jnp.int32, self.size)[::-1]

    @jax.jit
    def states_to_numbers(self, states: Array) -> Array:
        local_numbers = self.local_index.states_to_numbers(states, dtype=jnp.int32)
        return local_numbers @ self._basis

    @jax.jit
    def numbers_to_states(self, numbers: Array) -> Array:
        local_numbers = (numbers[..., None] // self._basis) % self.local_size
        return self.local_index.numbers_to_states(local_numbers)

    @jax.jit
    def all_states(self) -> Array:
        return self.numbers_to_states(jnp.arange(self.n_states, dtype=jnp.int32))

    @property
    def is_indexable(self) -> bool:
        return is_indexable(self.n_states)
