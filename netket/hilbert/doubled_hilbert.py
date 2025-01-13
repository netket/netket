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
from functools import wraps

import jax.numpy as jnp

from netket.utils.dispatch import parametric
from netket.utils.types import Array, DType

from .abstract_hilbert import AbstractHilbert
from .discrete_hilbert import DiscreteHilbert
from .index import is_indexable


@parametric
class DoubledHilbert(DiscreteHilbert):
    r"""
    Superoperatorial hilbert space for states living in the tensorised state
    :math:`\hat{H}\otimes \hat{H}`, encoded according to Choi's isomorphism.
    """

    def __init__(self, hilb: AbstractHilbert):
        r"""
        Superoperatorial hilbert space for states living in the tensorised
        state :math:`\hat{H}\otimes \hat{H}`, encoded according to Choi's
        isomorphism.

        Args:
            hilb: the Hilbert space H.

        Examples:
            Simple superoperatorial hilbert space for few spins.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=5,n_dim=2,pbc=True)
            >>> hi = nk.hilbert.Spin(N=3, s=0.5)
            >>> hi2 = nk.hilbert.DoubledHilbert(hi)
            >>> print(hi2.size)
            6
        """
        self.physical = hilb

        super().__init__(shape=hilb.shape * 2)

    @property
    @wraps(DiscreteHilbert.size)
    def size(self):
        return 2 * self.physical.size

    @property
    @wraps(DiscreteHilbert.shape)
    def shape(self):
        return self._shape

    @property
    @wraps(DiscreteHilbert.is_finite)
    def is_finite(self):
        return self.physical.is_finite

    @property
    @wraps(DiscreteHilbert.is_indexable)
    def is_indexable(self) -> bool:
        """Whether the space can be indexed with an integer"""
        n = self.physical.n_states
        return self.physical.is_indexable and is_indexable([n, n])

    @property
    def local_size(self):
        """The local size of the Hilbert space, if defined on the Physical
        Hilbert space.
        """
        return self.physical.local_size

    @property
    @wraps(DiscreteHilbert.all_states)
    def local_states(self):
        return self.physical.local_states

    @property
    @wraps(DiscreteHilbert.constrained)
    def constrained(self):
        return self.physical.constrained

    def size_at_index(self, i):
        r"""Size of the local degrees of freedom for the i-th variable.

        Args:
            i: The index of the desired site

        Returns:
            The number of degrees of freedom at that site
        """
        return self.physical.size_at_index(
            i if i < self.physical.size else i - self.physical.size
        )

    def states_at_index(self, i):
        r"""A list of discrete local quantum numbers at the site i.
        If the local states are infinitely many, None is returned.

        Args:
            i: The index of the desired site.

        Returns:
            A list of values or None if there are infinitely many.
        """
        return self.physical.states_at_index(
            i if i < self.physical.size else i - self.physical.size
        )

    @property
    def size_physical(self):
        """
        The size of the physical Hilbert space. In general
        this should be half of the size of the doubled Hilbert space.
        """
        return self.physical.size

    @property
    def n_states(self):
        """
        The total number of states in the Hilbert space. This should be
        the square of the number of states in the physical Hilbert space.
        """
        return self.physical.n_states**2

    def _numbers_to_states(self, numbers):
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

        dim = self.physical.n_states
        left, right = jnp.divmod(numbers, dim)

        out_l = self.physical.numbers_to_states(left)
        out_r = self.physical.numbers_to_states(right)
        return jnp.concatenate([out_l, out_r], axis=-1)

    def _states_to_numbers(self, states):
        # !!! WARNING
        # See note above in numbers_to_states

        n = self.physical.size
        dim = self.physical.n_states

        _out_l = self.physical._states_to_numbers(states[:, 0:n])
        _out_r = self.physical._states_to_numbers(states[:, n : 2 * n])
        return _out_l * dim + _out_r

    def states_to_local_indices(self, x):
        return self.physical.states_to_local_indices(x)

    def local_indices_to_states(self, x: Array, dtype: DType = None):
        return self.physical.local_indices_to_states(x, dtype=dtype)

    def __repr__(self):
        return f"DoubledHilbert({self.physical})"

    @property
    def _attrs(self):
        return (self.physical,)
