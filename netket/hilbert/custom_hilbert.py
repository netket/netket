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

from typing import Optional, Callable


from netket.utils import StaticRange

from .homogeneous import HomogeneousHilbert


class CustomHilbert(HomogeneousHilbert):
    r"""A custom hilbert space with discrete local quantum numbers."""

    def __init__(
        self,
        local_states: Optional[StaticRange],
        N: int = 1,
        constraint_fn: Optional[Callable] = None,
    ):
        r"""
        Constructs a new ``CustomHilbert`` given a list of eigenvalues of the states and
        a number of sites, or modes, within this hilbert space.

        Args:
            local_states: :class:`~netket.utils.StaticRange` object describing the
                numbers used to encode the local degree of freedom of this Hilbert
                Space.
            N: Number of modes in this hilbert space (default 1).
            constraint_fn: A function specifying constraints on the quantum numbers.
                Given a batch of quantum numbers it should return a vector
                of bools specifying whether those states are valid or not.

        The :class:`netket.utils.StaticRange` object works like a standard `range`
        object and is used to define the valid configurations of the local degrees
        of freedom.

        For example, the :class:`~netket.utils.StaticRange` of a Fock Hilbert space
        is constructed as

        .. code-block:: python

            >>> import netket as nk
            >>> n_max = 10
            >>> nk.utils.StaticRange(start=0, step=1, length=n_max)
            StaticRange(start=0, step=1, length=10, dtype=int64)

        and the range of a Spin-1/2 Hilbert space is constructed as:

        .. code-block:: python

            >>> import netket as nk
            >>> n_max = 10
            >>> nk.utils.StaticRange(start=-1, step=2, length=2)
            StaticRange(start=-1, step=2, length=2, dtype=int64)


        Examples:
            Simple custom hilbert space.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
            >>> local_states = nk.utils.StaticRange(start=-2.0, step=1.0, length=4)
            >>> hi = nk.hilbert.CustomHilbert(local_states=local_states, N=100)
            >>> print(hi.size)
            100
        """
        super().__init__(local_states, N, constraint_fn)

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        if not self.constrained:
            if self.local_states == other.local_states:
                return CustomHilbert(self._local_states, self.size + other.size)

        return NotImplemented
