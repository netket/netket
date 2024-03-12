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

from fractions import Fraction
from typing import Optional, Union

import numpy as np

from netket.utils import StaticRange

from .homogeneous import HomogeneousHilbert

from .index.constraints import SumConstraint


def _check_total_sz(total_sz, S, size):
    if total_sz is None:
        return

    local_size = 2 * S + 1

    m = round(2 * total_sz)
    if np.abs(m) > size * (2 * S):
        raise ValueError(
            "Cannot fix the total magnetization: 2|M| cannot " "exceed Nspins."
        )

    # If half-integer spins (1/2, 3/2)
    if local_size % 2 == 0:
        # Check that the total magnetization is odd if odd spins or even if even
        # number of spins
        if (size + m) % 2 != 0:
            raise ValueError(
                "Cannot fix the total magnetization: Nspins + 2*totalSz must be even."
            )
    # else if full-integer (S=1,2)
    else:
        if m % 2 != 0:
            raise ValueError(
                "Cannot fix the total magnetization to a half-integer number"
            )


class Spin(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local spin states."""

    def __init__(
        self,
        s: float,
        N: int = 1,
        total_sz: Optional[float] = None,
    ):
        r"""Hilbert space obtained as tensor product of local spin states.

        Args:
           s: Spin at each site. Must be integer or half-integer.
           N: Number of sites (default=1)
           total_sz: If given, constrains the total spin of system to a particular
                value.

        Examples:
           Simple spin hilbert space.

           >>> import netket as nk
           >>> hi = nk.hilbert.Spin(s=1/2, N=4)
           >>> print(hi.size)
           4
        """
        local_size = round(2 * s + 1)
        local_states = np.empty(local_size)

        assert int(2 * s + 1) == local_size
        local_states = StaticRange(
            1 - local_size, 2, local_size, dtype=np.int8 if local_size < 2**7 else int
        )

        _check_total_sz(total_sz, s, N)
        if total_sz is not None:
            constraints = SumConstraint(round(2 * total_sz))
        else:
            constraints = None

        self._total_sz = total_sz
        self._s = s

        super().__init__(local_states, N, constraints)

    def __pow__(self, n):
        if not self.constrained:
            return Spin(self._s, self.size * n)

        return NotImplemented

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        if self._s == other._s:
            if not self.constrained and not other.constrained:
                return Spin(s=self._s, N=self.size + other.size)

        return NotImplemented

    def ptrace(self, sites: Union[int, list]) -> Optional["Spin"]:
        if isinstance(sites, int):
            sites = [sites]

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        if self._total_sz is not None:
            raise TypeError(
                "Cannot take the partial trace with a total magnetization constraint."
            )

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            return Spin(s=self._s, N=self.size - Nsites)

    def __repr__(self):
        total_sz = f", total_sz={self._total_sz}" if self._total_sz is not None else ""
        return f"Spin(s={Fraction(self._s)}{total_sz}, N={self.size})"

    @property
    def _attrs(self):
        return (self.size, self._s, self._total_sz)
