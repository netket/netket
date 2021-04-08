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

import numpy as np

from . import AbstractGraph
from netket.utils.semigroup import SemiGroup


@dataclass(frozen=True)
class SymmGroup(SemiGroup):
    """
    Collection of symmetry operations acting on the sites of a graph
    (graph automorphisms).
    """

    graph: AbstractGraph
    """Underlying graph"""

    def __post_init__(self):
        super().__post_init__()
        myhash = hash((super().__hash__(), hash(self.graph)))
        object.__setattr__(self, "_SymmGroup__hash", myhash)

    def __matmul__(self, other):
        if isinstance(other, SymmGroup) and self.graph != other.graph:
            raise ValueError("Incompatible groups (underlying graph must be identical)")

        return SymmGroup(super().__matmul__(other).elems, self.graph)

    def to_array(self):
        """
        Convert the abstract group operations to an array of permutation indicies,
        such that (for :code:`G = self`)::
            V = np.array(G.graph.nodes())
            assert np.all(G(V) == V[..., G.to_array()])
        """
        return self.__call__(np.arange(self.graph.n_nodes))

    def __array__(self, dtype=None):
        return np.asarray(self.to_array(), dtype=dtype)

    def num_elements(self):
        return len(self)

    def remove_duplicates(self):
        """
        Returns a new :code:`SymmGroup` with duplicate elements (that is, elements which
        act on :code:`self.graph` identically) removed.
        """
        _, unique_indices = np.unique(self.to_array(), axis=0, return_index=True)
        return SymmGroup([self.elems[i] for i in sorted(unique_indices)], self.graph)

    def inverse(self):

        rm_dup = self.remove_duplicates()

        n_elem = rm_dup.num_elements()
        inverse = np.zeros([n_elem], dtype=int)
        sq = rm_dup.__matmul__(rm_dup).to_array()
        is_iden = np.where(~(sq - np.arange(sq.shape[-1])).any(axis=1))[0]

        inverse[is_iden // n_elem] = is_iden % n_elem

        return SymmGroup([self.elems[i] for i in inverse], self.graph)

    def group_algebra(self):

        group = self.remove_duplicates()
        n_elem = group.num_elements()
        group_algebra = np.zeros([n_elem, n_elem], dtype=int)
        inverse = self.inverse()
        comp = inverse.__matmul__(group).to_array()

        for n, elem in enumerate(group.to_array()):

            is_iden = np.where(~(comp - elem).any(axis=1))[0]

            group_algebra[is_iden % n_elem, is_iden // n_elem] = n

        return group_algebra.ravel()

    @property
    def shape(self):
        """Tuple `(<# of group elements>, <# of graph nodes>)`,
        same as :code:`self.to_array().shape`."""
        return (len(self), self.graph.n_nodes)

    def __hash__(self):
        return self.__hash

    def __repr__(self):
        return super().__repr__()
