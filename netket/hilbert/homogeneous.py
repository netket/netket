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

from numbers import Real

import numpy as np
from numba import jit

from .discrete_hilbert import DiscreteHilbert
from .hilbert_index import HilbertIndex


@jit(nopython=True)
def _gen_to_bare_numbers(conditions):
    return np.nonzero(conditions)[0]


@jit(nopython=True)
def _to_constrained_numbers_kernel(bare_numbers, numbers):
    found = np.searchsorted(bare_numbers, numbers)
    if np.max(found) >= bare_numbers.shape[0]:
        raise RuntimeError("The required state does not satisfy the given constraints.")
    return found


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

        self._size = N

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

        self._has_constraint = constraint_fn is not None
        self._constraint_fn = constraint_fn

        self._hilbert_index = None

        shape = tuple(self._local_size for _ in range(self.size))
        super().__init__(shape=shape)

    @property
    def size(self) -> int:
        r"""The total number number of degrees of freedom."""
        return self._size

    @property
    def local_size(self) -> int:
        r"""Size of the local degrees of freedom that make the total hilbert space."""
        return self._local_size

    def size_at_index(self, i: int) -> int:
        return self.local_size

    @property
    def local_states(self) -> Optional[List[float]]:
        r"""A list of discreet local quantum numbers.
        If the local states are infinitely many, None is returned."""
        return self._local_states

    def states_at_index(self, i: int):
        return self.local_states

    @property
    def n_states(self) -> int:
        r"""The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""

        hind = self._get_hilbert_index()

        if not self._has_constraint:
            return hind.n_states
        else:
            return self._bare_numbers.shape[0]

    @property
    def is_finite(self) -> bool:
        r"""Whether the local hilbert space is finite."""
        return self._is_finite

    @property
    def constrained(self) -> bool:
        r"""Returns True if the hilbert space is constrained."""
        return self._has_constraint

    def _numbers_to_states(self, numbers: np.ndarray, out: np.ndarray) -> np.ndarray:
        hind = self._get_hilbert_index()
        return hind.numbers_to_states(self._to_bare_numbers(numbers), out)

    def _states_to_numbers(self, states, out):
        hind = self._get_hilbert_index()

        hind.states_to_numbers(states, out)

        if self._has_constraint:
            out[:] = _to_constrained_numbers_kernel(
                self._bare_numbers,
                out,
            )

        return out

    def _get_hilbert_index(self):
        if self._hilbert_index is None:
            if not self.is_indexable:
                raise RuntimeError("The hilbert space is too large to be indexed.")

            self._hilbert_index = HilbertIndex(
                np.asarray(self.local_states, dtype=np.float64), self.size
            )

            if self._has_constraint:
                self._bare_numbers = _gen_to_bare_numbers(
                    self._constraint_fn(self._hilbert_index.all_states())
                )
            else:
                self._bare_numbers = np.empty(0, dtype=np.intp)

        return self._hilbert_index

    def _to_bare_numbers(self, numbers):
        if self._constraint_fn is None:
            return numbers
        else:
            return self._bare_numbers[numbers]

    def __repr__(self):
        constr = (
            ", has_constraint={}".format(self._has_constraint)
            if self._has_constraint
            else ""
        )

        clsname = type(self).__name__
        return f"{clsname}(local_size={self._local_size}, N={self.size}{constr})"

    @property
    def _attrs(self):
        return (
            self.size,
            self.local_size,
            self._local_states_frozen,
            self._has_constraint,
            self._constraint_fn,
        )
