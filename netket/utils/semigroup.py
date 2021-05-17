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


from abc import ABC, abstractmethod
from dataclasses import dataclass
import itertools
from typing import Callable, List

from plum import dispatch

import numpy as np

from netket.utils import struct
from netket.utils import HashableArray
from netket.utils.types import Array, DType, Shape


class ElementBase(ABC):
    @abstractmethod
    def __call__(self, arg):
        pass

    def __matmul__(self, other):
        return product(self, other)


class Element(ElementBase):
    pass


@dataclass(frozen=True)
class Identity(ElementBase):
    def __call__(self, arg):
        return arg

    def __repr__(self):
        return "Id()"


@dispatch
def product(a: Identity, _: Identity):
    return a


@dispatch
def product(_: Identity, b: Element):
    return b


@dispatch
def product(a: Element, _: Identity):
    return a


@dataclass(frozen=True)
class Composite(Element):
    left: Element
    right: Element

    def __call__(self, arg):
        return self.left(self.right(arg))

    def __repr__(self):
        return f"{self.left} @ {self.right}"


@dispatch
def product(a: Element, b: Element):
    return Composite(a, b)


@dispatch
def product(ab: Composite, c: Element):
    bc = product(ab.right, c)
    if isinstance(bc, Composite):
        return Composite(ab, c)
    else:
        return Composite(ab.left, bc)


@dispatch
def product(a: Element, bc: Composite):
    ab = product(a, bc.left)
    if isinstance(ab, Composite):
        return Composite(a, bc)
    else:
        return Composite(ab, bc.right)


@dispatch
def product(ab: Composite, cd: Composite):
    bc = product(ab.right, cd.left)
    if isinstance(bc, Composite):
        return Composite(ab, cd)
    else:
        return Composite(ab.left, Composite(bc, cd.right))


@dataclass(frozen=True)
class NamedElement(Element):
    name: str
    action: Callable
    info: str = ""

    def __call__(self, arg):
        return self.action(arg)

    def __repr__(self):
        return f"{self.name}({self.info})"


class Permutation(Element):
    def __init__(self, permutation: Array):
        self.permutation = HashableArray(np.asarray(permutation))

    def __call__(self, x):
        return x[..., self.permutation]

    def __hash__(self):
        return hash(self.permutation)

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return self.permutation == other.permutation
        else:
            return False

    def __repr__(self):
        return f"Permutation({self.permutation})"

    def __array__(self, dtype: DType = None):
        return np.asarray(self.permutation, dtype)


@dispatch
def product(p: Permutation, q: Permutation):
    return Permutation(p(q.permutation))


@dataclass(frozen=True)
class SemiGroup:
    elems: List[Element]

    def __post_init__(self):
        # manually assign self.__hash == ... for frozen dataclass,
        # see https://docs.python.org/3/library/dataclasses.html#frozen-instances
        myhash = hash(tuple(hash(x) for x in self.elems))
        object.__setattr__(self, "_SemiGroup__hash", myhash)

    def __matmul__(self, other):
        """
        Direct product of this group with `other`.
        """
        return SemiGroup(
            elems=[a @ b for a, b in itertools.product(self.elems, other.elems)],
        )

    def __call__(self, initial):
        """
        Apply all group elements to all entries of `initial` along the last axis.
        """
        initial = np.asarray(initial)
        return np.array([np.apply_along_axis(elem, -1, initial) for elem in self.elems])

    def __getitem__(self, i):
        return self.elems[i]

    def __hash__(self):
        return self.__hash  # pylint: disable=no-member

    def __iter__(self):
        return iter(self.elems)

    def __len__(self):
        return len(self.elems)

    def __repr__(self):
        if len(self.elems) > 31:
            elems = list(map(repr, self.elems[:15])) + list(map(repr, self.elems[-15:]))
            elems.insert(15, "...")
        else:
            elems = map(repr, self.elems)
        return type(self).__name__ + "(\n  {}\n)".format(",\n  ".join(elems))


@struct.dataclass()
class PermutationGroup(SemiGroup):
    """
    Collection of permutation operations acting on sequences of length :code:`degree`.
    Note that the group elements do not need to be of type :ref:`netket.utils.semigroup.Permutation`,
    only act as such on a sequence when called.

    Note that this class can contain elements that are distinct as objects (e.g.,
    :code:`Identity()` and :code:`Translation((0,))`) but have identical action.
    Those can be removed by calling :code:`remove_duplicates`.
    """

    degree: int
    """Number of elements the permutations act on."""

    def __post_init__(self):
        super().__post_init__()
        myhash = hash((super().__hash__(), hash(self.degree)))
        object.__setattr__(self, "_PermutationGroup__hash", myhash)

    def __matmul__(self, other) -> "PermutationGroup":
        if not isinstance(other, PermutationGroup):
            raise ValueError(
                "Incompatible groups (`PermutationGroup` and something else)"
            )
        if isinstance(other, PermutationGroup) and self.degree != other.degree:
            raise ValueError(
                "Incompatible groups (`PermutationGroup`s of different degree)"
            )

        return PermutationGroup(super().__matmul__(other).elems, self.degree)

    def to_array(self) -> Array:
        """
        Convert the abstract group operations to an array of permutation indices,
        such that the `i`-th row contains the indices corresponding to the `i`-th group
        element (i.e., `self.to_array()[i, j]` is :math:`g_i(j)`) and
        (for :code:`G = self`)::
            V = np.arange(G.degree)
            assert np.all(G(V) == V[..., G.to_array()])
        """
        return self.__call__(np.arange(self.degree))

    def __array__(self, dtype=None) -> Array:
        return np.asarray(self.to_array(), dtype=dtype)

    def remove_duplicates(self, *, return_inverse=False) -> "PermutationGroup":
        """
        Returns a new :code:`PermutationGroup` with duplicate elements (that is, elements which
        represent identical permutations) removed.

        Arguments:
            return_inverse: If True, also return indices to reconstruct the original
                group from the result.

        Returns:
            group: the permutation group with duplicate elements removed.
            return_inverse: Indices to reconstruct the original group from the result.
                Only returned if `return_inverse` is True.
        """
        result = np.unique(
            self.to_array(),
            axis=0,
            return_index=True,
            return_inverse=return_inverse,
        )
        group = PermutationGroup(
            [self.elems[i] for i in sorted(result[1])], self.degree
        )
        if return_inverse:
            return group, result[2]
        else:
            return group

    @struct.property_cached
    def inverse(self) -> Array:
        """
        Returns the indices of the inverse of each element.

        If :code:`g = self[idx_g]` and :code:`h = self[self.inverse[idx_g]]`, then
        :code:`gh = product(g, h)` will act as the identity on any sequence,
        i.e., :code:`np.all(gh(seq) == seq)`.
        """
        perm_array = self.to_array()
        n_symm = len(perm_array)
        inverse = np.zeros([n_symm], dtype=int)
        for i, perm1 in enumerate(perm_array):
            for j, perm2 in enumerate(perm_array):
                perm_sq = perm1[perm2]
                if np.all(perm_sq == np.arange(len(perm_sq))):
                    inverse[i] = j

        return inverse

    @struct.property_cached
    def product_table(self) -> Array:
        """
        Returns a table of indices corresponding to :math:`g^{-1} h` over the group.

        That is, if :code:`g = self[idx_g]', :code:`h = self[idx_h]`, and
        :code:`idx_u = self.product_table[idx_g, idx_h]`, then :code:`self[idx_u]`
        corresponds to :math:`u = g^{-1} h`.
        """
        perms = self.to_array()
        inverse = perms[self.inverse].squeeze()
        n_symm = len(perms)
        product_table = np.zeros([n_symm, n_symm], dtype=int)

        inv_t = inverse.transpose()
        perms_t = perms.transpose()
        inv_elements = perms_t[inv_t].reshape(-1, n_symm * n_symm).transpose()

        perms = [HashableArray(element) for element in perms]
        inv_perms = [HashableArray(element) for element in inv_elements]

        inverse_index_mapping = {element: index for index, element in enumerate(perms)}

        inds = [
            (index, inverse_index_mapping[element])
            for index, element in enumerate(inv_perms)
            if element in inverse_index_mapping
        ]

        inds = np.asarray(inds)

        product_table[inds[:, 0] // n_symm, inds[:, 0] % n_symm] = inds[:, 1]

        return product_table

    @property
    def shape(self) -> Shape:
        """Tuple `(<# of group elements>, <degree>)`, same as :code:`self.to_array().shape`."""
        return (len(self), self.degree)

    def __hash__(self):
        # pylint: disable=no-member
        return self.__hash
