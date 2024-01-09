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

from functools import lru_cache

from .unconstrained import UnconstrainedHilbertIndex

import numpy as np


# This function has exponential runtime in self.size, so we cache it in order to
# only compute it once.
# TODO: distribute over MPI... chose better chunk size
@lru_cache(maxsize=5)
def compute_constrained_to_bare_conversion_table(
    hilbert_index, constraint_fn, *, chunk_size: int = 100000
):
    """
    Computes the conversion table that converts the 'constrained' indices
    of an hilbert space to bare indices, so that routines generating
    only values in an unconstrained space can be used.

    This function operates on blocks of `chunk_size` states at a time in order
    to lower the memory cost. The default chunk size has been chosen by instinct
    and is likely wrong.
    """

    n_chunks = int(np.ceil(hilbert_index.n_states / chunk_size))
    bare_number_chunks = []
    for i in range(n_chunks):
        id_start = chunk_size * i
        id_end = np.minimum(chunk_size * (i + 1), hilbert_index.n_states)
        ids = np.arange(id_start, id_end)

        states = hilbert_index.numbers_to_states(ids)
        is_constrained = constraint_fn(states)
        (chunk_bare_number,) = np.nonzero(is_constrained)
        bare_number_chunks.append(chunk_bare_number + id_start)

    return np.concatenate(bare_number_chunks)


class ConstrainedHilbertIndex:
    def __init__(self, local_states, size, constraint_fun):
        self._unconstrained_index = UnconstrainedHilbertIndex(local_states, size)
        self._constraint_fn = constraint_fun

        self.__bare_numbers = None

    @property
    def size(self) -> int:
        return self._unconstrained_index.size

    @property
    def n_states(self):
        return self._bare_numbers.shape[0]

    @property
    def local_states(self):
        return self._unconstrained_index._local_states

    @property
    def local_size(self) -> int:
        return self._unconstrained_index.local_size

    @property
    def _bare_numbers(self) -> np.ndarray:
        """
        Returns the conversion table between indices in the constrained space and
        the corresponding unconstrained space.
        """
        if self.__bare_numbers is None:
            self.__bare_numbers = compute_constrained_to_bare_conversion_table(
                self._unconstrained_index, self._constraint_fn
            )

        return self.__bare_numbers

    def states_to_numbers(self, states, out=None):
        self._unconstrained_index.states_to_numbers(states, out)

        out[:] = np.searchsorted(self._bare_numbers, out)

        if np.max(out, initial=0) >= self.n_states:
            raise RuntimeError(
                "The required state does not satisfy " "the given constraints."
            )

        return out

    def numbers_to_states(self, numbers, out=None):
        if numbers.ndim != 1:
            raise RuntimeError("Invalid input shape, expecting a 1d array.")

        # convert to original space
        numbers = self._bare_numbers[numbers]

        if out is None:
            out = np.empty((numbers.shape[0], self.size))

        for i in range(numbers.shape[0]):
            out[i] = self._unconstrained_index.number_to_state(numbers[i])

        return out

    def all_states(self, out=None):
        return self.numbers_to_states(np.arange(self.n_states), out=out)
