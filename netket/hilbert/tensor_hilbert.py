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

from collections.abc import Iterable

from abc import ABC

import numpy as np

from .abstract_hilbert import AbstractHilbert


class TensorHilbert(ABC):
    r"""Abstract base class for the tensor product of several sub-spaces,
    representing the space.

    This class can also be used to construct the correct type of TensorHilbert
    subclass given the input types: if all input types are Generic hilbert spaces,
    `TensorGeneralHilbert` will be constructed, while if they all are `DiscreteHilbert`
    a `TensorDiscreteHilbert` will be created.

    In general you should not construct this object directly, but you should
    simply multiply different hilbert spaces together. In this case, Python's
    `*` operator will be interpreted as a tensor product.

    This is an abstract mixing class that should be inherited from, together
    with another class that inherits from `AbstractHilbert`.
    """

    def __new__(cls, *args, **kwargs):
        # This logic overrides the constructor, such that if someone tries to
        # construct this class directly by calling `TensorHilbert(...)`
        # it will construct either a DiscreteHilbert or TensorDiscreteHilbert
        from .tensor_hilbert_discrete import TensorDiscreteHilbert, DiscreteHilbert

        if cls is TensorHilbert:
            if all(isinstance(hi, DiscreteHilbert) for hi in args):
                cls = TensorDiscreteHilbert
            else:
                cls = TensorGenericHilbert

        return super().__new__(cls)

    def __init__(self, hilb_spaces: Iterable[AbstractHilbert], *args, **kwargs):
        r"""Constructs a tensor Hilbert space.

        Args:
            *hilb: An iterable object containing at least 1 hilbert space.
        """
        # Flatten "TensorHilberts" found inside hilb_spaces
        _hilb_spaces_flat: list[AbstractHilbert] = []
        for hi in hilb_spaces:
            if isinstance(hi, TensorHilbert):
                _hilb_spaces_flat.extend(hi.subspaces)
            else:
                _hilb_spaces_flat.append(hi)
        hilb_spaces = _hilb_spaces_flat

        self._hilbert_spaces = tuple(hilb_spaces)
        self._n_hilbert_spaces = len(hilb_spaces)
        self._hilbert_i = np.concatenate(
            [[i for _ in range(hi.size)] for (i, hi) in enumerate(hilb_spaces)]
        )

        self._sizes = tuple([hi.size for hi in hilb_spaces])
        self._cum_sizes = np.cumsum(self._sizes)
        self._cum_indices = np.concatenate([[0], self._cum_sizes])
        self._size = sum(self._sizes)
        self._delta_indices_i = np.array(
            [self._cum_indices[i] for i in self._hilbert_i]
        )
        super().__init__(
            *args, **kwargs
        )  # forwards all unused arguments so that this class is a mixin.

    @property
    def size(self) -> int:
        return self._size

    @property
    def subspaces(self) -> tuple[AbstractHilbert, ...]:
        """Tuplec ontaining all the subspaces of this tensor product of
        Hilbert spaces.
        """
        return self._hilbert_spaces

    def _sub_index(self, i: int) -> int:
        """Internal function computing the subspace index for the given site
        i.

        Arguments:
            i: Index of a site in :math:`[0, N)`.

        Returns:
            The index `j` such that self.subspaces[j] is the Hilbert space
            containing site `i`.
        """
        for j, sz in enumerate(self._cum_sizes):
            if i < sz:
                return j

    @property
    def _attrs(self):
        return self._hilbert_spaces

    def ptrace(self, sites: int | list) -> AbstractHilbert | None:
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

    def __repr__(self):
        if len(self._hilbert_spaces) == 1:
            return f"{type(self).__name__}({self._hilbert_spaces[0]})"

        _str = f"{self._hilbert_spaces[0]}"
        for hi in self._hilbert_spaces[1:]:
            _str += f"⊗{hi}"

        return _str

    def __mul__(self, other):
        spaces_l = self._hilbert_spaces[:-1]
        space_center_l = self._hilbert_spaces[-1]

        if isinstance(other, TensorHilbert):
            space_center_r = other._hilbert_spaces[0]
            spaces_r = other._hilbert_spaces[1:]
        else:
            space_center_r = other
            spaces_r = tuple()

        # Attempt to 'merge' the two spaces at the interface.
        spaces_center = space_center_l * space_center_r
        if isinstance(spaces_center, TensorHilbert):
            spaces_center = (space_center_l, space_center_r)
        else:
            spaces_center = (spaces_center,)

        return TensorHilbert(*spaces_l, *spaces_center, *spaces_r)


class TensorGenericHilbert(TensorHilbert, AbstractHilbert):
    def __init__(self, *hilb_spaces: AbstractHilbert):
        if not all(isinstance(hi, AbstractHilbert) for hi in hilb_spaces):
            raise TypeError(
                "Arguments to TensorHilbert must all be subtypes of "
                "AbstractHilbert. However the types are:\n\n"
                f"{list(type(hi) for hi in hilb_spaces)}\n"
            )
        super().__init__(hilb_spaces)
