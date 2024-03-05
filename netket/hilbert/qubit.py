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

from typing import Optional, Union

import numpy as np

from netket.utils import StaticRange

from .homogeneous import HomogeneousHilbert


class Qubit(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local qubit states."""

    def __init__(self, N: int = 1):
        r"""Initializes a qubit hilbert space.

        Args:
            N: Number of qubits.

        Examples:
            Simple spin hilbert space.

            >>> from netket.hilbert import Qubit
            >>> hi = Qubit(N=100)
            >>> print(hi.size)
            100
        """
        super().__init__(StaticRange(0, 1, 2, dtype=np.int8), N)

    def __pow__(self, n):
        return Qubit(self.size * n)

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        return Qubit(self.size + other.size)

    def ptrace(self, sites: Union[int, list]) -> Optional["Qubit"]:
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
        return f"Qubit(N={self.size})"

    @property
    def _attrs(self):
        return (self.size,)
