# Copyright 2023 The NetKet Authors - All rights reserved.
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

import numpy as np

from netket.utils import struct
from netket.utils.types import Array, Scalar

max_states = np.iinfo(np.int32).max
"""int: Maximum number of states that can be indexed"""


def is_indexable(shape: tuple[Scalar] | Scalar) -> bool:
    """
    Returns whether a discrete Hilbert space of shape `shape` is
    indexable (i.e., its total number of states is below the maximum).
    """
    # cast to float to avoid
    # TypeError: loop of ufunc does not support argument 0 of type int which has no callable log method
    # when shape contains ints larger than the max int64
    shape = np.asarray(shape, dtype=np.float64)
    log_max = np.log(max_states)
    return np.sum(np.log(shape)) <= log_max


class HilbertIndex(struct.Pytree):
    @property
    def n_states(self) -> int:
        """Returns the size of the hilbert space."""
        return NotImplemented  # pragma: no cover

    @property
    def is_indexable(self) -> bool:
        """Whether the index can be indexed with an integer"""
        return NotImplemented  # pragma: no cover

    def states_to_numbers(self, states: Array) -> Array:
        """Given a Batch of N states of size M, returns an array
        of np.int32 integers used to numerate those states.
        """
        raise NotImplementedError  # pragma: no cover

    def numbers_to_states(self, numbers: Array) -> Array:
        """Given a list of N integers, returns a batch of N states of size M
        of corresponding to those states. This function is the inverse
        of `states_to_numbers`.
        """
        raise NotImplementedError  # pragma: no cover

    def all_states(self) -> Array:
        """Should behave as `self.numbers_to_states(range(self.n_states))`
        but might be optimised for iterating across the full hilbert
        space.
        """
        raise NotImplementedError  # pragma: no cover
