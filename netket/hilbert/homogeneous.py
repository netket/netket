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

from typing import Optional, List, Callable
from functools import lru_cache

from numbers import Real

import numpy as np

from netket.errors import HilbertIndexingDuringTracingError, concrete_or_error

from .discrete_hilbert import DiscreteHilbert
from .hilbert_index import HilbertIndex


# This function has exponential runtime in self.size, so we cache it in order to
# only compute it once.
# TODO: distribute over MPI... chose better chunk size
@lru_cache(maxsize=5)
def compute_constrained_to_bare_conversion_table(self, *, chunk_size: int = 100000):
    """
    Computes the conversion table that converts the 'constrained' indices
    of an hilbert space to bare indices, so that routines generating
    only values in an unconstrained space can be used.

    This function operates on blocks of `chunk_size` states at a time in order
    to lower the memory cost. The default chunk size has been chosen by instinct
    and is likely wrong.
    """
    n_chunks = int(np.ceil(self._hilbert_index.n_states / chunk_size))
    bare_number_chunks = []
    for i in range(n_chunks):
        id_start = chunk_size * i
        id_end = np.minimum(chunk_size * (i + 1), self._hilbert_index.n_states)
        ids = np.arange(id_start, id_end)

        states = self._hilbert_index.numbers_to_states(ids)
        is_constrained = self._constraint_fn(states)
        (chunk_bare_number,) = np.nonzero(is_constrained)
        bare_number_chunks.append(chunk_bare_number + id_start)

    return np.concatenate(bare_number_chunks)


class HomogeneousHilbert(DiscreteHilbert):
    r"""The Abstract base class for homogeneous hilbert spaces.

    This class should only be subclassed and should not be instantiated directly.
    """

    def __init__(
        self,
        local_states: Optional[List[Real]],
        N: int = 1,
        constraint_fn: Optional[Callable] = None,
    ):
        r"""
        Constructs a new ``HomogeneousHilbert`` given a list of eigenvalues of the
        states and a number of sites, or modes, within this hilbert space.

        This method should only be called from the subclasses `__init__` method.

        Args:
            local_states (list or None): Eigenvalues of the states. If the allowed
                states are an infinite number, None should be passed as an argument.
            N: Number of modes in this hilbert space (default 1).
            constraint_fn: A function specifying constraints on the quantum numbers.
                Given a batch of quantum numbers it should return a vector of bools
                specifying whether those states are valid or not.
        """
        assert isinstance(N, int)

        self._is_finite = local_states is not None

        if self._is_finite:
            self._local_states = np.asarray(local_states)
            assert self._local_states.ndim == 1
            self._local_size = self._local_states.shape[0]
            self._local_states = self._local_states.tolist()
            self._local_states_frozen = frozenset(self._local_states)
        else:
            self._local_states = None
            self._local_states_frozen = None
            self._local_size = np.iinfo(np.intp).max

        self._constraint_fn = constraint_fn

        self.__hilbert_index = None
        self.__bare_numbers = None

        shape = tuple(self._local_size for _ in range(N))
        super().__init__(shape=shape)

    @property
    def size(self) -> int:
        r"""The total number number of degrees of freedom."""
        return len(self.shape)

    @property
    def local_size(self) -> int:
        r"""Size of the local degrees of freedom that make the total hilbert space."""
        return self._local_size

    def size_at_index(self, i: int) -> int:
        return self.local_size

    @property
    def local_states(self) -> Optional[List[float]]:
        r"""A list of discrete local quantum numbers.
        If the local states are infinitely many, None is returned."""
        return self._local_states

    def states_at_index(self, i: int):
        return self.local_states

    @property
    def n_states(self) -> int:
        r"""The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""
        if not self.constrained:
            return self._hilbert_index.n_states
        else:
            return self._bare_numbers.shape[0]

    @property
    def is_finite(self) -> bool:
        r"""Whether the local hilbert space is finite."""
        return self._is_finite

    @property
    def constrained(self) -> bool:
        r"""Returns True if the hilbert space is constrained."""
        return self._constraint_fn is not None

    def _numbers_to_states(self, numbers: np.ndarray, out: np.ndarray) -> np.ndarray:

        numbers = concrete_or_error(
            np.asarray, numbers, HilbertIndexingDuringTracingError
        )

        if self.constrained:
            numbers = self._bare_numbers[numbers]

        return self._hilbert_index.numbers_to_states(numbers, out)

    def _states_to_numbers(self, states: np.ndarray, out: np.ndarray):

        states = concrete_or_error(
            np.asarray, states, HilbertIndexingDuringTracingError
        )

        self._hilbert_index.states_to_numbers(states, out)

        if self.constrained:
            out[:] = np.searchsorted(self._bare_numbers, out)

            if np.max(out) >= self.n_states:
                raise RuntimeError(
                    "The required state does not satisfy " "the given constraints."
                )

        return out

    @property
    def _hilbert_index(self) -> HilbertIndex:
        """
        Returns the `HilbertIndex` object, which is a numba jitclass used to convert
        integers to states and vice-versa.
        """
        if self.__hilbert_index is None:
            if not self.is_indexable:
                raise RuntimeError("The hilbert space is too large to be indexed.")

            self.__hilbert_index = HilbertIndex(
                np.asarray(self.local_states, dtype=np.float64), self.size
            )

        return self.__hilbert_index

    @property
    def _bare_numbers(self) -> np.ndarray:
        """
        Returns the conversion table between indices in the constrained space and
        the corresponding unconstrained space.
        """
        if not self.constrained:
            return None

        if self.__bare_numbers is None:
            self.__bare_numbers = compute_constrained_to_bare_conversion_table(self)

        return self.__bare_numbers

    def __repr__(self):
        constr = f", constrained={self.constrained}" if self.constrained else ""

        clsname = type(self).__name__
        return f"{clsname}(local_size={self._local_size}, N={self.size}{constr})"

    @property
    def _attrs(self):
        return (
            self.size,
            self.local_size,
            self._local_states_frozen,
            self.constrained,
            self._constraint_fn,
        )
