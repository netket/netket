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

from dataclasses import dataclass
import itertools
from typing import Any, Callable, List

from plum import dispatch

import numpy as np

from netket.utils import HashableArray
from netket.utils.types import Array, DType


class ElementBase(Callable):
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
        return self.__hash

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


@dataclass(frozen=True)
class PermutationGroup(SemiGroup):
    """
    Collection of permutation operations acting on sequences of :code:`degree` elements.
    Note that the elements do not need to have type :ref:`netket.utils.semigroup.Permutation`,
    only act as such on a sequence when called.

    Note that this class can contain elements that are distinct as objects (e.g., :code:`Identity()` and
    :code:`Translation((0,))`) but identical action. Those can be removed by calling :code:`remove_duplicates`.
    """

    degree: int
    """Number of elements the permutations act on"""

    def __post_init__(self):
        super().__post_init__()
        myhash = hash((super().__hash__(), hash(self.degree)))
        object.__setattr__(self, "_PermutationGroup__hash", myhash)

        object.__setattr__(self, "_inverse", None)
        object.__setattr__(self, "_product_table", None)

    def __matmul__(self, other):
        if isinstance(other, PermutationGroup) and self.degree != other.degree:
            raise ValueError(
                "Incompatible groups (`PermutationGroup`s of different degree)"
            )

        return PermutationGroup(super().__matmul__(other).elems, self.degree)

    def to_array(self):
        """
        Convert the abstract group operations to an array of permutation indicie s,
        such that (for :code:`G = self`)::
            V = np.array(G.graph.nodes())
            assert np.all(G(V) == V[..., G.to_array()])
        """
        return self.__call__(np.arange(self.degree))

    def __array__(self, dtype=None):
        return np.asarray(self.to_array(), dtype=dtype)

    def remove_duplicates(self, *, return_inverse=False):
        """
        Returns a new :code:`PermutationGroup` with duplicate elements (that is, elements which
        represent identical permutations) removed.

        Arguments:
            return_inverse: If True, also return indices to reconstruct the original
                group from the result.

        Returns:
            symm_group: the symmetry group with duplicates removed.
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

    def __inverse(self):
        """
        Returns indices of the involution of the PermutationGroup where the each element is the inverse of
        the original symmetry element. If :code:`g = self[element]` and :code:`h = self[self.inverse()][element]`,
        then :code:`gh = product(g, h)` will act as the identity on the sites of the graph, i.e., :code:`np.all(gh(sites) == sites)`.
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

    def __product_table(self):
        """
        Returns a product table over the group where the columns use the involution
        of the group. If :code:`g = self[self.inverse()[element]]', :code:`h = self[element2]`
        and code:`u = self[product_table()[element,element2]], we are
        solving the equation u = gh
        """
        perms = self.to_array()
        inverse = perms[self.inverse()].squeeze()
        n_symm = len(perms)
        product_table = np.zeros([n_symm, n_symm], dtype=int)

        inv_t = inverse.transpose()
        perms_t = perms.transpose()
        inv_elements = perms_t[inv_t].reshape(-1, n_symm * n_symm).transpose()

        perms = [HashableArray(element) for element in perms]
        inv_perms = [HashableArray(element) for element in inv_elements]

        inverse_index_mapping = {element: index for index, element in perms}

        inds = [
            (index, inverse_index_mapping[element])
            for index, element in enumerate(inv_perms)
            if element in inverse_index_mapping
        ]

        inds = np.asarray(inds)

        product_table[inds[:, 0] // n_symm, inds[:, 0] % n_symm] = inds[:, 1]

        return product_table

    def inverse(self):
        if self._inverse is None:
            object.__setattr__(self, "_inverse", self.__inverse())

        return self._inverse

    def product_table(self):
        if self._product_table is None:
            object.__setattr__(self, "_product_table", self.__product_table())

        return self._product_table

    @property
    def shape(self):
        """Tuple `(<# of group elements>, <degree>)`, same as :code:`self.to_array().shape`."""
        return (len(self), self.degree)

    def __hash__(self):
        return self.__hash
