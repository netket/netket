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


@dataclass
class SymmGroup(SemiGroup):
    """
    Collection of symmetry operations acting on the sites of a graph
    (graph automorphisms).
    """

    graph: AbstractGraph
    """Underlying graph"""

    def __matmul__(self, other):
        if isinstance(other, SymmGroup) and self.graph is not other.graph:
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

    def __hash__(self):
        return hash((super().__hash__(), hash(self.graph)))

    @property
    def shape(self):
        """Tuple `(<# of group elements>, <# of graph nodes>)`,
        same as :code:`self.to_array().shape`."""
        return (len(self), self.graph.n_nodes)

    def __repr__(self):
        return super().__repr__()
