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

from functools import partial
import inspect
import itertools
from typing import Any, Callable, List, Optional, Tuple

from dataclasses import dataclass
import multipledispatch
import numpy as np

namespace = {}
dispatch = partial(multipledispatch.dispatch, namespace=namespace)


class ElementBase(Callable):
    pass


class Element(ElementBase):
    pass


@dataclass
class Identity(ElementBase):
    def __call__(self, arg):
        return arg

    def __repr__(self):
        return "Id()"


@dispatch(Identity, Identity)
def product(a: Identity, _: Identity):
    return a


@dispatch(Identity, Element)
def product(_: Identity, b: Element):
    return b


@dispatch(Element, Identity)
def product(a: Element, _: Identity):
    return a


@dataclass
class Composite(Element):
    left: Element
    right: Element

    def __call__(self, arg):
        return self.left(self.right(arg))

    def __repr__(self):
        return f"{self.left} @ {self.right}"


@dispatch(Element, Element)
def product(a: Element, b: Element):
    return Composite(a, b)


@dispatch(Composite, Element)
def product(ab: Composite, c: Element):
    bc = product(ab.right, c)
    if isinstance(bc, Composite):
        return Composite(ab, c)
    else:
        return Composite(ab.left, bc)


@dispatch(Element, Composite)
def product(a: Element, bc: Composite):
    ab = product(a, bc.left)
    if isinstance(ab, Composite):
        return Composite(a, bc)
    else:
        return Composite(ab, bc.right)


@dispatch(Composite, Composite)
def product(ab: Composite, cd: Composite):
    bc = product(ab.right, cd.left)
    if isinstance(bc, Composite):
        return Composite(ab, cd)
    else:
        return Composite(ab.left, Composite(bc, cd.right))


@dataclass
class NamedElement(Element):
    name: str
    action: Callable
    info: str = ""

    def __call__(self, arg):
        return self.action(arg)

    def __repr__(self):
        return f"{self.name}({self.info})"


@dataclass
class SemiGroup:
    elems: List[Element]

    def __matmul__(self, other):
        """
        Direct product of this group with `other`.
        """
        return SemiGroup(
            elems=[
                product(a, b) for a, b in itertools.product(self.elems, other.elems)
            ],
        )

    def __getitem__(self, i):
        return self.elems[i]

    def __len__(self):
        return len(self.elems)

    def __call__(self, initial):
        """
        Apply all group elements to all entries of `initial` along the last axis.
        """
        initial = np.asarray(initial)
        return np.array([np.apply_along_axis(elem, -1, initial) for elem in self.elems])

    def __repr__(self):
        if len(self.elems) > 31:
            elems = list(map(repr, self.elems[:15])) + list(map(repr, self.elems[-15:]))
            elems.insert(15, "...")
        else:
            elems = map(repr, self.elems)
        return type(self).__name__ + "(\n  {}\n)".format(",\n  ".join(elems))
