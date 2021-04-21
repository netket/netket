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
from netket.utils.types import Array
from netket.utils import HashableArray


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

    def remove_duplicates(self, *, return_inverse=False):
        """
        Returns a new :code:`SymmGroup` with duplicate elements (that is, elements which
        act on :code:`self.graph` identically) removed.

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
        group = SymmGroup([self.elems[i] for i in sorted(result[1])], self.graph)
        if return_inverse:
            return group, result[2]
        else:
            return group

    def inverse(self):
        """
        Returns reordered SymmGroup where the each element is the inverse of
        the original symmetry element. If :code:`g = self[element]` and :code:`h = self[self.inverse()[element]]`,
        then :code:`gh = product(g, h)` will act as the identity on the sites of the graph, i.e., :code:`np.all(gh(sites) == sites)`.

        """

        automorphisms = self.to_array()
        n_symm = len(automorphisms)
        inverse = np.zeros([n_symm], dtype=int)
        automorphisms = np.array(automorphisms)
        for i, perm1 in enumerate(automorphisms):
            for j, perm2 in enumerate(automorphisms):
                perm_sq = perm1[perm2]
                if np.all(perm_sq == np.arange(len(perm_sq))):
                    inverse[i] = j

        return inverse

    def product_table(self):
        """
        Returns a product table over the group where the columns use the involution
        of the group. If :code:`g = self[self.inverse()[element]]', :code:`h = self[element2]`
        and code:`u = self[product_table()[element,element2]], we are
        solving the equation u = gh

        """

        automorphisms = self.to_array()
        n_symm = len(automorphisms)
        inverse = self.to_array()[self.inverse()]
        product_table = np.zeros([n_symm, n_symm], dtype=int)

        inv_t = inverse.transpose()
        auto_t = automorphisms.transpose()
        inv_auto = auto_t[inv_t].reshape(-1, n_symm * n_symm).transpose()

        hash_auto = {
            hash(element.tobytes()): index
            for index, element in enumerate(automorphisms)
        }

        inds = [
            (index, hash_auto[hash(element.tobytes())])
            for index, element in enumerate(inv_auto)
            if hash(element.tobytes()) in hash_auto
        ]

        inds = np.asarray(inds)

        product_table[inds[:, 0] // n_symm, inds[:, 0] % n_symm] = inds[:, 1]

        return product_table

    @property
    def shape(self):
        """Tuple `(<# of group elements>, <# of graph nodes>)`,
        same as :code:`self.to_array().shape`."""
        return (len(self), self.graph.n_nodes)

    def __hash__(self):
        return self.__hash

    def __repr__(self):
        return super().__repr__()
