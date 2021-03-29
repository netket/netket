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

from typing import List, Optional, Tuple

import jax
from jax import numpy as jnp
import numpy as np

from .abstract_hilbert import AbstractHilbert


class DoubledHilbert(AbstractHilbert):
    r"""Superoperatorial hilbert space for states living in the
    tensorised state H\otimes H, encoded according to Choi's isomorphism."""

    def __init__(self, hilb):
        r"""Superoperatorial hilbert space for states living in the
           tensorised state H\otimes H, encoded according to Choi's isomorphism.

        Args:
            hilb: the hilbrt space H.

        Examples:
            Simple superoperatorial hilbert space for few spins.

           >>> from netket.hilbert import Spin, DoubledHilbert
           >>> g = Hypercube(length=5,n_dim=2,pbc=True)
           >>> hi = Spin(N=3, s=0.5)
           >>> hi2 = DoubledHilbert(hi)
           >>> print(hi2.size)
           50
        """
        self.physical = hilb
        self._size = 2 * hilb.size

        self._shape = hilb.shape * 2
        super().__init__()

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def is_discrete(self):
        return self.physical.is_discrete

    @property
    def is_finite(self):
        return self.physical.is_finite

    @property
    def local_size(self):
        return self.physical.local_size

    @property
    def local_states(self):
        return self.physical.local_states

    def size_at_index(self, i: int) -> int:
        return self.physical.size_at_index(
            i if i < self.physical.size else i - self.physical.size
        )

    def states_at_index(self, i: int) -> Optional[List[float]]:
        r"""A list of discrete local quantum numbers at the site i.
        If the local states are infinitely many, None is returned."""
        return self.physical.states_at_index(
            i if i < self.physical.size else i - self.physical.size
        )

    @property
    def size_physical(self):
        return self.physical.size

    @property
    def n_states(self):
        return self.physical.n_states ** 2

    def _numbers_to_states(self, numbers, out):
        # !!! WARNING
        # This code assumes that states are stored in a MSB
        # (Most Significant Bit) format.
        # We assume that the rightmost-half indexes the LSBs
        # and the leftmost-half indexes the MSBs
        # HilbertIndex-generated states respect this, as they are:
        # 0 -> [0,0,0,0]
        # 1 -> [0,0,0,1]
        # 2 -> [0,0,1,0]
        # etc...

        n = self.physical.size
        dim = self.physical.n_states
        left, right = np.divmod(numbers, dim)

        self.physical.numbers_to_states(left, out=out[:, 0:n])
        self.physical.numbers_to_states(right, out=out[:, n : 2 * n])

        return out

    def _states_to_numbers(self, states, out):
        # !!! WARNING
        # See note above in numbers_to_states

        n = self.physical.size
        dim = self.physical.n_states

        self.physical._states_to_numbers(states[:, 0:n], out=out)
        _out_l = out * dim

        self.physical._states_to_numbers(states[:, n : 2 * n], out=out)
        out += _out_l

        return out

    def __repr__(self):
        return "DoubledHilbert({})".format(self.physical)

    @property
    def _attrs(self):
        return (self.physical,)

    def _random_state_legacy(self, size=None, *, out=None, rgen=None):
        if isinstance(size, int):
            size = (size,)
        shape = (*size, self.size) if size is not None else (self.size,)

        if out is None:
            out = np.empty(shape=shape)

        n = self.size_physical

        self.physical.random_state(out=out[..., :n], size=size, rgen=rgen)
        self.physical.random_state(out=out[..., n:], size=size, rgen=rgen)

        return out
