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
from abc import ABC
from dataclasses import dataclass
from typing import List

import numpy as np

from netket.utils.types import Array
from netket.utils.dispatch import dispatch


class Element(ABC):
    """Base element of a FiniteSemiGroup.

    Every element must define at least a dispatch rule
    to determine how it is applied to a ket as follows:

    @nk.utils.dispatch.dispatch
    def product(A: MyElement, b: nk.utils.types.Array):
        return A*b

    An element can also define dispatch rules to combine
    itself with other elements
    """

    def __call__(self, ket):
        return product(self, ket)

    def __matmul__(self, other):
        return product(self, other)


@dataclass(frozen=True)
class Identity(Element):
    """The identity transformation."""

    def __repr__(self):
        return "Id()"


@dispatch
def product(_: Identity, ket: Array):
    return ket


@dispatch
def product(a: Identity, _: Identity):  # noqa: F811
    return a


@dispatch
def product(_: Identity, b: Element):  # noqa: F811
    return b


@dispatch
def product(a: Element, _: Identity):  # noqa: F811
    return a


@dataclass(frozen=True)
class Composite(Element):
    """
    Composition of two elements of a finite group, representing
    `left@right`
    """

    left: Element
    right: Element

    def __repr__(self):
        return f"{self.left} @ {self.right}"


@dispatch
def product(a: Composite, ket: Array):  # noqa: F811
    return a.left(a.right(ket))


@dispatch
def product(a: Element, b: Element):  # noqa: F811
    return Composite(a, b)


@dispatch
def product(ab: Composite, c: Element):  # noqa: F811
    bc = product(ab.right, c)
    if isinstance(bc, Composite):
        return Composite(ab, c)
    else:
        return Composite(ab.left, bc)


@dispatch
def product(a: Element, bc: Composite):  # noqa: F811
    ab = product(a, bc.left)
    if isinstance(ab, Composite):
        return Composite(a, bc)
    else:
        return Composite(ab, bc.right)


@dispatch
def product(ab: Composite, cd: Composite):  # noqa: F811
    bc = product(ab.right, cd.left)
    if isinstance(bc, Composite):
        return Composite(ab, cd)
    else:
        return Composite(ab.left, Composite(bc, cd.right))


@dataclass(frozen=True)
class FiniteSemiGroup:
    elems: List[Element]

    def __post_init__(self):
        # manually assign self.__hash == ... for frozen dataclass,
        # see https://docs.python.org/3/library/dataclasses.html#frozen-instances
        myhash = hash(tuple(hash(x) for x in self.elems))
        object.__setattr__(self, "_FiniteSemiGroup__hash", myhash)

    def __matmul__(self, other):
        """
        Cartesian product of this semigroup with `other`.
        """
        return product(self, other)

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


@dispatch
def product(A: FiniteSemiGroup, B: FiniteSemiGroup):  # noqa: F811
    return FiniteSemiGroup(
        elems=[a @ b for a, b in itertools.product(A.elems, B.elems)],
    )
