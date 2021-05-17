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

# Ignore false-positives for redefined `product` functions:
# pylint: disable=function-redefined

import numpy as np
from dataclasses import dataclass

from .semigroup import SemiGroup, Element, Identity
from netket.utils import struct, HashableArray
from netket.utils.types import Array, DType, Shape
from typing import Tuple


@dataclass(frozen=True)
# @struct.dataclass
class Group(SemiGroup):
    """
    Collection of Elements expected to satisfy group axioms.
    Unlike SemiGroup, product tables, conjugacy classes, etc. can be calculated.

    Group elements can be implemented in any way, as long as a subclass of Group
    is able to implement their action. Subclasses must implement a :code:`_canonical()`
    method that returns an array of integers for each acceptable Element such that
    two Elements are considered equal iff the corresponding matrices are equal.

    """

    def __post_init__(self):
        super().__post_init__()

    def __matmul__(self, other) -> "Group":
        if not isinstance(other, Group):
            raise ValueError("`Group`s can only be multiplied with other `Group`s")
        return Group(super().__matmul__(other).elems)

    def _canonical(self, x: Element) -> Array:
        """
        Canonical form of :code:`Element`s, used for equality testing (i.e., two :code:`Element`s
        `x,y` are deemed equal iff :code:`_canonical(x) == _canonical(y)`.
        Must be overridden in subclasses

        Arguments:
            x: an `Element`

        Returns:
            the canonical form as a numpy.ndarray of integers
        """
        raise NotImplementedError

    def _canonical_array(self) -> Array:
        """
        Lists the canonical forms returned by `_canonical` as rows of a 2D array.
        """
        return np.array([self._canonical(x).flatten() for x in self.elems])

    def _canonical_lookup(self) -> dict:
        """
        Creates a lookup table from canonical forms to index in `self.elems`
        """
        return {
            HashableArray(self._canonical(element)): index
            for index, element in enumerate(self.elems)
        }

    def remove_duplicates(self, *, return_inverse=False) -> "Group":
        """
        Returns a new :code:`Group` with duplicate elements (that is, elements with
        identical canonical forms) removed.

        Arguments:
            return_inverse: If True, also return indices to reconstruct the original
                group from the result.

        Returns:
            group: the group with duplicate elements removed.
            inverse: Indices to reconstruct the original group from the result.
                Only returned if `return_inverse` is True.
        """
        result = np.unique(
            self._canonical_array(),
            axis=0,
            return_index=True,
            return_inverse=return_inverse,
        )
        group = Group([self.elems[i] for i in sorted(result[1])])
        if return_inverse:
            return group, result[2]
        else:
            return group

    # @struct.property_cached
    @property
    def inverse(self) -> Array:
        """
        Returns the indices of the inverse of each element.
        If :code:`g = self[idx_g]` and :code:`h = self[self.inverse[idx_g]]`, then
        :code:`gh = product(g, h)` is equivalent to :code:`Identity()`
        """
        canonical_identity = self._canonical(Identity())
        inverse = np.zeros(len(self.elems), dtype=int)

        for i, e1 in enumerate(elems):
            for j, e2 in enumerate(elems):
                prod = e1 @ e2
                if np.all(self._canonical(prod) == canonical_identity):
                    inverse[i] = j

        return inverse

    # @struct.property_cached
    @property
    def product_table(self) -> Array:
        """
        Returns a table of indices corresponding to :math:`g^{-1} h` over the group.
        That is, if :code:`g = self[idx_g]', :code:`h = self[idx_h]`, and
        :code:`idx_u = self.product_table[idx_g, idx_h]`, then :code:`self[idx_u]`
        corresponds to :math:`u = g^{-1} h`.
        """
        n_symm = len(self)
        product_table = np.zeros([n_symm, n_symm], dtype=int)

        lookup = self._canonical_lookup()

        for i, e1 in enumerate(self.elems[self.inverse]):
            for j, e2 in enumerate(self.elems):
                prod = e1 @ e2
                product_table[i, j] = lookup[HashableArray(self._canonical(prod))]

        return product_table

    # @struct.property_cached
    @property
    def conjugacy_table(self) -> Array:
        """
        Returns a table of conjugates: if `g = self[idx_g]` and `h = self[idx_h]`,
        then `self[self.conjugacy_table[idx_g,idx_h]]` is :math:`h^{-1}gh`.
        """
        col_index = np.arange(len(self))[np.newaxis, :]
        # exploits that h^{-1}gh = (g^{-1} h)^{-1} h
        return self.product_table[self.product_table, col_index]

    # @struct.property_cached
    @property
    def conjugacy_classes(self) -> Tuple:
        """
        Finds the conjugacy classes of the group.

        Returns:
            classes: a boolean array, each row indicating the elements that belong to one conjugacy class
            representatives: the lowest-indexed member of each conjugacy class
            inverse: the conjugacy class index of every group element
        """
        row_index = np.arange(len(self))[:, np.newaxis]

        # if is_conjugate[i,j] is True, self[i] and self[j] are in the same class
        is_conjugate = np.full((len(self), len(self)), False)
        is_conjugate[row_index, self.conjugacy_table] = True

        classes, representatives, inverse = np.unique(
            is_conjugate, axis=0, return_index=True, return_inverse=True
        )

        # Usually self[0] == Identity(), which is its own class
        # This class is listed last by the lexicographic order used by np.unique
        # so we reverse it to get a more conventional layout
        classes = classes[::-1]
        representatives = representatives[::-1]
        inverse = (representatives.size - 1) - inverse

        return classes, representatives, inverse

    def __hash__(self):
        return super().__hash__()
