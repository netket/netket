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

from typing import Optional, Union, List

import numpy as np

from netket.graph import AbstractGraph

from .homogeneous import HomogeneousHilbert
from ._deprecations import graph_to_N_depwarn


class Qubit(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local qubit states."""

    def __init__(self, N: int = 1, graph: Optional[AbstractGraph] = None):
        r"""Initializes a qubit hilbert space.

        Args:
            N: Number of qubits.
            graph: (deprecated) a graph from which to extract the number of sites.

        Examples:
            Simple spin hilbert space.

            >>> from netket.hilbert import Qubit
            >>> hi = Qubit(N=100)
            >>> print(hi.size)
            100
        """
        N = graph_to_N_depwarn(N=N, graph=graph)

        super().__init__([0.0, 1.0], N)

    def states_to_local_indices(self, x):
        return x.astype(np.int32)

    def __pow__(self, n):
        return Qubit(self.size * n)

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        return Qubit(self.size + other.size)

    def ptrace(self, sites: Union[int, List]) -> Optional["Qubit"]:
        if isinstance(sites, int):
            sites = [sites]

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            return Qubit(N=self.size - Nsites)

    def __repr__(self):
        return "Qubit(N={})".format(self._size)

    @property
    def _attrs(self):
        return (self.size,)
