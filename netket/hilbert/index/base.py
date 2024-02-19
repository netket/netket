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

from typing import Protocol

from netket.utils.types import Array


class HilbertIndex(Protocol):
    @property
    def n_states(self) -> int:
        """Returns the size of the hilbert space."""

    def states_to_numbers(self, states: Array) -> Array:
        """Given a Batch of N states of size M, returns an array
        of np.int32 or np.int64 integers used to numerate those states.
        """

    def numbers_to_states(self, numbers: Array) -> Array:
        """Given a list of N integers, returns a batch of N states of size M
        of corresponding to those states. This function is the inverse
        of `states_to_numbers`.
        """

    def all_states(self) -> Array:
        """Should behave as `self.numbers_to_states(range(self.n_states))`
        but might be optimised for iterating across the full hilbert
        space.
        """
