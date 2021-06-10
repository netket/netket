# Copyright 2021 The NetKet Authors - All rights reserved.
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
from numba.experimental import jitclass
from numba import int64, float64


spec = [
    ("_local_states", float64[:]),
    ("_local_size", int64),
    ("_size", int64),
    ("_basis", int64[:]),
]


@jitclass(spec)
class HilbertIndex:
    def __init__(self, local_states, size):
        self._local_states = np.sort(local_states)
        self._local_size = len(self._local_states)
        self._size = size

        self._basis = np.zeros(size, dtype=np.int64)
        ba = 1
        for s in range(size):
            self._basis[s] = ba
            ba *= self._local_size

    def _local_state_number(self, x):
        return np.searchsorted(self.local_states, x)

    @property
    def n_states(self):
        return self._local_size ** self._size

    @property
    def local_states(self):
        return self._local_states

    def state_to_number(self, state):
        if state.ndim != 1:
            raise RuntimeError("Invalid input shape, expecting a 1d array.")

        # Converts a vector of quantum numbers into the unique integer identifier
        number = 0

        for i in range(self._size):
            number += (
                self._local_state_number(state[self._size - i - 1]) * self._basis[i]
            )

        return number

    def number_to_state(self, number, out=None):

        if out is None:
            out = np.empty(self._size)
        # else:
        #     assert out.size == self._size

        out.fill(self._local_states[0])

        ip = number
        k = self._size - 1
        while ip > 0:
            out[k] = self._local_states[ip % self._local_size]
            ip = ip // self._local_size
            k -= 1

        return out

    def states_to_numbers(self, states, out=None):
        if states.ndim != 2:
            raise RuntimeError("Invalid input shape, expecting a 2d array.")

        if out is None:
            out = np.empty(states.shape[0], np.int64)
        # else:
        #     assert out.size == states.shape[0]

        for i in range(states.shape[0]):
            out[i] = 0
            for j in range(self._size):
                out[i] += (
                    self._local_state_number(states[i, self._size - j - 1])
                    * self._basis[j]
                )
        return out

    def numbers_to_states(self, numbers, out=None):
        if numbers.ndim != 1:
            raise RuntimeError("Invalid input shape, expecting a 1d array.")

        if out is None:
            out = np.empty((numbers.shape[0], self._size))
        # else:
        #     assert out.shape == (numbers.shape[0], self._size)

        for i in range(numbers.shape[0]):
            out[i] = self.number_to_state(numbers[i])

        return out

    def all_states(self, out=None):
        if out is None:
            out = np.empty((self.n_states, self._size))

        for i in range(self.n_states):
            self.number_to_state(i, out[i])
        return out
