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

import itertools
from typing import Optional

import numpy as np

from netket.utils import HashableArray, struct
from netket.utils.types import Array, DType, Shape
from netket.utils.dispatch import dispatch

from ._group import FiniteGroup
from ._semigroup import Element


class Permutation(Element):
    def __init__(self, permutation: Array, name: Optional[str] = None):
        r"""
        Creates a `Permutation` from an array of preimages of :code:`range(N)`

        Arguments:
            permutation: 1D array listing :math:`g^{-1}(x)` for all :math:`0\le x < N`
                (i.e., `V[permutation]` permutes the elements of `V` as desired)
            name: optional, custom name for the permutation

        Returns:
            a `Permutation` object encoding the same permutation
        """
        self.permutation = HashableArray(np.asarray(permutation))
        self.__name = name

    def __hash__(self):
        return hash(self.permutation)

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self.permutation == other.permutation
        else:
            return False

    @property
    def _name(self):
        return self.__name

    def __repr__(self):
        if self._name is not None:
            return self._name
        else:
            return f"Permutation({np.asarray(self).tolist()})"

    def __array__(self, dtype: DType = None):
        return np.asarray(self.permutation, dtype)


@dispatch
def product(p: Permutation, x: Array):
    return x[..., p.permutation]


@dispatch
def product(p: Permutation, q: Permutation):  # noqa: F811
    name = None if p._name is None and q._name is None else f"{p} @ {q}"
    return Permutation(p(np.asarray(q)), name)


@struct.dataclass
class PermutationGroup(FiniteGroup):
    r"""
    Collection of permutation operations acting on sequences of length :code:`degree`.

    Group elements need not all be of type :ref:`netket.utils.group.Permutation`,
    only act as such on a sequence when called.

    The class can contain elements that are distinct as objects (e.g.,
    :code:`Identity()` and :code:`Translation((0,))`) but have identical action.
    Those can be removed by calling :code:`remove_duplicates`.
    """

    degree: int
    """Number of elements the permutations act on."""

    def __hash__(self):
        return super().__hash__()

    def _canonical(self, x: Element) -> Array:
        return x(np.arange(self.degree, dtype=int))

    def to_array(self) -> Array:
        r"""
        Convert the abstract group operations to an array of permutation indices,
        such that the `i`-th row contains the indices corresponding to the `i`-th group
        element. That is, `self.to_array()[i, j]` is :math:`g_i^{-1}(j)`) and
        (for :code:`G = self`)::
            V = np.arange(G.degree)
            assert np.all(G(V) == V[..., G.to_array()])
        """
        return self._canonical_array()

    def __array__(self, dtype=None) -> Array:
        return np.asarray(self.to_array(), dtype=dtype)

    def remove_duplicates(self, *, return_inverse=False) -> "PermutationGroup":
        r"""
        Returns a new :code:`PermutationGroup` with duplicate elements (that is,
        elements which represent identical permutations) removed.

        Arguments:
            return_inverse: If True, also return indices to reconstruct the original
                group from the result.

        Returns:
            group: the permutation group with duplicate elements removed.
            return_inverse: Indices to reconstruct the original group from the result.
                Only returned if `return_inverse` is True.
        """
        if return_inverse:
            group, inverse = super().remove_duplicates(return_inverse=True)
        else:
            group = super().remove_duplicates(return_inverse=False)

        pgroup = PermutationGroup(group.elems, self.degree)

        if return_inverse:
            return pgroup, inverse
        else:
            return pgroup

    @struct.property_cached
    def inverse(self) -> Array:
        try:
            lookup = self._canonical_lookup()
            inverses = []
            for perm in self.to_array():
                invperm = np.argsort(perm)
                inverses.append(lookup[HashableArray(invperm)])

            return np.asarray(inverses, dtype=int)
        except KeyError as err:
            raise RuntimeError(
                "PermutationGroup does not contain the inverse of all elements"
            ) from err

    @struct.property_cached
    def product_table(self) -> Array:
        perms = self.to_array()
        inverse = perms[self.inverse].squeeze()
        n_symm = len(perms)
        lookup = np.unique(np.column_stack((perms, np.arange(len(self)))), axis=0)

        product_table = np.zeros([n_symm, n_symm], dtype=int)
        for i, g_inv in enumerate(inverse):
            row_perms = perms[:, g_inv]
            row_perms = np.unique(
                np.column_stack((row_perms, np.arange(len(self)))), axis=0
            )
            # row_perms should be a permutation of perms, so identical after sorting
            if np.any(row_perms[:, :-1] != lookup[:, :-1]):
                raise RuntimeError(
                    "PermutationGroup is not closed under multiplication"
                )
            # match elements in row_perms to group indices
            product_table[i, row_perms[:, -1]] = lookup[:, -1]

        return product_table

    @property
    def shape(self) -> Shape:
        r"""
        Tuple `(<# of group elements>, <degree>)`.

        Equivalent to :code:`self.to_array().shape`.
        """
        return (len(self), self.degree)


@dispatch
def product(A: PermutationGroup, B: PermutationGroup):  # noqa: F811
    if A.degree != B.degree:
        raise ValueError(
            "Incompatible groups (`PermutationGroup`s of different degree)"
        )
    return PermutationGroup(
        elems=[a @ b for a, b in itertools.product(A.elems, B.elems)], degree=A.degree
    )
