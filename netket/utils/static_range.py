# Copyright 2022 The NetKet Authors - All rights reserved.
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

from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import struct
from netket.utils.types import DType
from netket.jax import canonicalize_dtypes


class Range(struct.Pytree):
    """
    An object representing a range similar to python's range, but that
    works with `jax.jit` and can be used within Numba-blocks.

    This range object can also be used to convert 'computational basis'
    configurations to integer indices ∈ [0,length].
    """

    start: Number = struct.field(pytree_node=False)
    step: Number = struct.field(pytree_node=False)
    length: int = struct.field(pytree_node=False)
    dtype: DType = struct.field(pytree_node=False)

    def __init__(self, start: Number, step: Number, length: int, dtype: DType = None):
        """
        Constructs a Static Range object.

        Args:
            start: Value of the first entry
            step: step between the entries
            length: Length of this range
        """
        dtype = canonicalize_dtypes(start, step, dtype=dtype)

        self.start = jnp.array(start, dtype=dtype)
        self.step = jnp.array(step, dtype=dtype)
        self.length = length

        self.dtype = dtype

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError
        return self.start + self.step * i

    def find(self, val):
        return int((val - self.start) / self.step)

    @jax.jit
    def states_to_numbers(self, x, dtype: DType = None):
        idx = (x - self.start) / self.step
        if dtype is not None:
            idx = idx.astype(dtype)
        return idx

    @jax.jit
    def numbers_to_states(self, i, dtype: DType = None):
        state = self.start + self.step * i
        if dtype is not None:
            state = state.astype(dtype)
        return state

    def flip_state(self, state):
        if not len(self) == 2:
            raise ValueError
        constant_sum = 2 * self.start + self.step
        return constant_sum - state

    def __array__(self, dtype=None):
        return self.start + np.arange(self.length, dtype=dtype) * self.step

    def __hash__(self):
        return hash(("StaticRange", self.start, self.step, self.length))

    def __eq__(self, o):
        if isinstance(o, Range):
            return (
                self.start == o.start
                and self.step == o.step
                and self.length == o.length
            )
        else:
            return self.__array__() == o

    def __repr__(self):
        return f"StaticRange(start={self.start}, step={self.step}, length={self.length}, dtype={self.dtype})"
