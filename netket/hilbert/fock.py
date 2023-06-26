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

from typing import List, Optional, Union
from functools import partial

import numpy as np
from numba import jit

from .homogeneous import HomogeneousHilbert

FOCK_MAX = np.iinfo(np.intp).max - 1
"""
Maximum number of particles in the fock space.
It is `maxvalue(np.int64)-1` because we use N+1 in several formulas
and it would overflow.
"""


@jit(nopython=True)
def _sum_constraint(x, n_particles):
    return np.sum(x, axis=1) == n_particles


class Fock(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local fock basis."""

    def __init__(
        self,
        n_max: Optional[int] = None,
        N: int = 1,
        n_particles: Optional[int] = None,
    ):
        r"""
        Constructs a new ``Boson`` given a maximum occupation number, number of sites
        and total number of bosons.

        Args:
          n_max: Maximum occupation for a site (inclusive). If None, the local
            occupation number is unbounded.
          N: number of bosonic modes (default = 1)
          n_particles: Constraint for the number of particles. If None, no constraint
            is imposed.

        Examples:
           Simple boson hilbert space.

           >>> from netket.hilbert import Fock
           >>> hi = Fock(n_max=5, n_particles=11, N=3)
           >>> print(hi.size)
           3
           >>> print(hi.n_states)
           15
        """
        self._n_max = n_max

        if n_particles is not None:
            n_particles = int(n_particles)
            if n_particles < 0:
                raise ValueError(
                    f"Number of particles must be >= 0, but received {n_particles}."
                )
            self._n_particles = n_particles

            if self._n_max is None:
                self._n_max = n_particles
            else:
                if self._n_max * N < n_particles:
                    raise Exception(
                        """The required total number of bosons is not compatible
                        with the given n_max."""
                    )

            constraints = partial(_sum_constraint, n_particles=n_particles)

        else:
            constraints = None
            self._n_particles = None

        if self._n_max is not None:
            # assert self._n_max > 0
            local_states = np.arange(self._n_max + 1)
        else:
            self._n_max = FOCK_MAX
            local_states = None

        super().__init__(local_states, N, constraints)

    @property
    def n_max(self) -> Optional[int]:
        r"""The maximum number of bosons per site, or None
        if the number is unconstrained."""
        return self._n_max

    @property
    def n_particles(self) -> Optional[int]:
        r"""The total number of particles, or None
        if the number is unconstrained."""
        return self._n_particles

    def __pow__(self, n) -> "Fock":
        if self.n_particles is None:
            return Fock(self.n_max, self.size * n)

        return NotImplemented

    def _mul_sametype_(self, other: "Fock") -> "Fock":
        assert type(self) == type(other)
        if self.n_max == other.n_max:
            if self._n_particles is None and other._n_particles is None:
                return Fock(self.n_max, N=self.size + other.size)

        return NotImplemented

    def ptrace(self, sites: Union[int, List]) -> Optional["Fock"]:
        if isinstance(sites, int):
            sites = [sites]

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        if self.n_particles is not None:
            raise TypeError(
                "Cannot take the partial trace with a total particles constraint."
            )

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            return Fock(self.n_max, N=self.size - Nsites)

    def __repr__(self):
        n_particles = (
            f", n_particles={self._n_particles}"
            if self._n_particles is not None
            else ""
        )
        nmax = self._n_max if self._n_max < FOCK_MAX else "FOCK_MAX"
        return f"Fock(n_max={nmax}{n_particles}, N={self.size})"

    def states_to_local_indices(self, x):
        return x.astype(np.int32)

    @property
    def _attrs(self):
        return (self.size, self._n_max, self._n_particles)
