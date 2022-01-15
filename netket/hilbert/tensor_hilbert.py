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

from typing import Optional, List, Union

import numpy as np

from .discrete_hilbert import DiscreteHilbert, _is_indexable


# TODO: Make parametric class
class TensorHilbert(DiscreteHilbert):
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
        Fock(n_max=3, N=1)*Spin(s=1/2, N=5)
        >>> type(_)
        <class 'netket.hilbert.tensor_hilbert.TensorHilbert'>

    """

    def __init__(self, *hilb_spaces: DiscreteHilbert):
        r"""Constructs a tensor Hilbert space

        Args:
            *hilb: An iterable object containing at least 1 hilbert space.
        """

        self._hilbert_spaces = hilb_spaces
        self._n_hilbert_spaces = len(hilb_spaces)
        self._hilbert_i = np.concatenate(
            [[i for _ in range(hi.size)] for (i, hi) in enumerate(hilb_spaces)]
        )

        self._sizes = tuple([hi.size for hi in hilb_spaces])
        shape = np.concatenate([hi.shape for hi in hilb_spaces])
        self._cum_sizes = np.cumsum(self._sizes)
        self._cum_indices = np.concatenate([[0], self._cum_sizes])
        self._size = sum(self._sizes)
        self._delta_indices_i = np.array(
            [self._cum_indices[i] for i in self._hilbert_i]
        )

        # pre-compute indexing data iff the tensor space is still indexable
        if all(hi.is_indexable for hi in hilb_spaces) and _is_indexable(shape):
            self._ns_states = [hi.n_states for hi in self._hilbert_spaces]
            self._ns_states_r = np.flip(self._ns_states)
            self._cum_ns_states = np.concatenate([[0], np.cumprod(self._ns_states)])
            self._cum_ns_states_r = np.flip(
                np.cumprod(np.concatenate([[1], np.flip(self._ns_states)]))[:-1]
            )
            self._n_states = np.prod(self._ns_states)

        super().__init__(shape=shape)

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_finite(self):
        return all([hi.is_finite for hi in self._hilbert_spaces])

    def _sub_index(self, i):
        for (j, sz) in enumerate(self._cum_sizes):
            if i < sz:
                return j

    # def size_at_index(self, i):
    #    # j = self._sub_index(i)
    #    # return self._hilbert_spaces[j].size_at_index(i-self._cum_indices[j-1])
    #    return self._hilbert_spaces[self._hilbert_i[i]].size_at_index(
    #        i - self._delta_indices_i[i]
    #    )

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

    def __repr__(self):
        _str = "{}".format(self._hilbert_spaces[0])
        for hi in self._hilbert_spaces[1:]:
            _str += "*{}".format(hi)

        return _str

    @property
    def _attrs(self):
        return self._hilbert_spaces

    def __mul__(self, other):
        spaces_l = self._hilbert_spaces[:-1]
        space_center_l = self._hilbert_spaces[-1]

        if isinstance(other, TensorHilbert):
            space_center_r = other._hilbert_spaces[0]
            spaces_r = other._hilbert_spaces[1:]
        else:
            space_center_r = other
            spaces_r = tuple()

        try:
            spaces_center = space_center_l * space_center_r
            if isinstance(spaces_center, TensorHilbert):
                spaces_center = (space_center_l, space_center_r)
            else:
                spaces_center = (spaces_center,)
        except Exception:
            spaces_center = (space_center_l, space_center_r)

        return TensorHilbert(*spaces_l, *spaces_center, *spaces_r)

    def ptrace(self, sites: Union[int, List]) -> Optional[DiscreteHilbert]:
        if isinstance(sites, int):
            sites = [sites]

        sites = np.sort(sites)

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            new_hilberts = []
            sz = 0
            for hilb in self._hilbert_spaces:
                sites_this_hilb = (
                    sites[np.logical_and(sites >= sz, sites < sz + hilb.size)] - sz
                )
                if len(sites_this_hilb) == 0:
                    new_hilberts.append(hilb)
                else:
                    ptraced_hilb = hilb.ptrace(sites_this_hilb)
                    if ptraced_hilb is not None:
                        new_hilberts.append(ptraced_hilb)
                sz += hilb.size

            if len(new_hilberts) == 0:
                return None
            elif len(new_hilberts) >= 1:
                hilb = new_hilberts[0]

                for h in new_hilberts[1:]:
                    hilb = hilb * h

                return hilb
