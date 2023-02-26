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
import jax.numpy as jnp

from .discrete_hilbert import DiscreteHilbert, _is_indexable
from .tensor_hilbert import TensorHilbert


class TensorDiscreteHilbert(TensorHilbert, DiscreteHilbert):
    r"""Tensor product of several Discrete sub-spaces, representing the space

    In general you should not construct this object directly, but you should
    simply multiply different hilbert spaces together. In this case, Python's
    `*` operator will be interpreted as a tensor product.

    This Hilbert can be used as a replacement anywhere a Uniform Hilbert space
    is not required.

    Examples:
        Couple a bosonic mode with spins

        >>> from netket.hilbert import Spin, Fock
        >>> Fock(3)*Spin(0.5, 5)
        Fock(n_max=3, N=1)⊗Spin(s=1/2, N=5)
        >>> type(_)
        <class 'netket.hilbert.tensor_hilbert.TensorHilbert'>

    """

    def __init__(self, *hilb_spaces: DiscreteHilbert):
        r"""Constructs a tensor Hilbert space

        Args:
            *hilb: An iterable object containing at least 1 hilbert space.
        """
        if not all(isinstance(hi, DiscreteHilbert) for hi in hilb_spaces):
            raise TypeError(
                "Arguments to TensorDiscreteHilbert must all be "
                "subtypes of DiscreteHilbert. However the types are:\n\n"
                f"{list(type(hi) for hi in hilb_spaces)}\n"
            )

        shape = np.concatenate([hi.shape for hi in hilb_spaces])

        super().__init__(hilb_spaces, shape=shape)

        # pre-compute indexing data iff the tensor space is still indexable
        if all(hi.is_indexable for hi in hilb_spaces) and _is_indexable(shape):
            self._ns_states = [hi.n_states for hi in self._hilbert_spaces]
            self._ns_states_r = np.flip(self._ns_states)
            self._cum_ns_states = np.concatenate([[0], np.cumprod(self._ns_states)])
            self._cum_ns_states_r = np.flip(
                np.cumprod(np.concatenate([[1], np.flip(self._ns_states)]))[:-1]
            )
            self._n_states = np.prod(self._ns_states)

    @property
    def is_finite(self):
        return all([hi.is_finite for hi in self._hilbert_spaces])

    def states_at_index(self, i):
        # j = self._sub_index(i)
        # return self._hilbert_spaces[j].states_at_index(i-self._cum_indices[j-1])
        return self._hilbert_spaces[self._hilbert_i[i]].states_at_index(
            i - self._delta_indices_i[i]
        )

    @property
    def n_states(self) -> int:
        if not self.is_indexable:
            raise RuntimeError("The hilbert space is too large to be indexed.")
        return self._n_states

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

        rem = numbers
        for (i, dim) in enumerate(self._ns_states_r):
            rem, loc_numbers = np.divmod(rem, dim)
            hi_i = self._n_hilbert_spaces - (i + 1)
            self._hilbert_spaces[hi_i].numbers_to_states(
                loc_numbers, out=out[:, self._cum_indices[hi_i] : self._cum_sizes[hi_i]]
            )

        return out

    def _states_to_numbers(self, states, out):
        out[:] = 0

        temp = out.copy()

        # !!! WARNING
        # See note above in numbers_to_states

        for (i, dim) in enumerate(self._cum_ns_states_r):
            self._hilbert_spaces[i].states_to_numbers(
                states[:, self._cum_indices[i] : self._cum_sizes[i]], out=temp
            )
            out += temp * dim

        return out

    def states_to_local_indices(self, x):
        out = jnp.empty_like(x, dtype=jnp.int32)
        for (i, hilb_i) in enumerate(self._hilbert_spaces):
            out = out.at[..., self._cum_indices[i] : self._cum_sizes[i]].set(
                hilb_i.states_to_local_indices(
                    x[..., self._cum_indices[i] : self._cum_sizes[i]]
                )
            )
        return out

    def __mul__(self, other):
        if not isinstance(other, DiscreteHilbert):
            return NotImplemented

        spaces_l = self._hilbert_spaces[:-1]
        space_center_l = self._hilbert_spaces[-1]

        if isinstance(other, TensorDiscreteHilbert):
            space_center_r = other._hilbert_spaces[0]
            spaces_r = other._hilbert_spaces[1:]
        else:
            space_center_r = other
            spaces_r = tuple()

        # Attempt to 'merge' the two spaces at the interface.
        spaces_center = space_center_l * space_center_r
        if isinstance(spaces_center, TensorDiscreteHilbert):
            spaces_center = (space_center_l, space_center_r)
        else:
            spaces_center = (spaces_center,)

        return TensorDiscreteHilbert(*spaces_l, *spaces_center, *spaces_r)
