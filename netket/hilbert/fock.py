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

from typing import Optional

import numpy as np

from netket.utils import StaticRange

from .homogeneous import HomogeneousHilbert
from .constraint import DiscreteHilbertConstraint, SumConstraint

FOCK_MAX = np.iinfo(np.intp).max - 1
"""
Maximum number of particles in the fock space.
It is `maxvalue(np.int64)-1` because we use N+1 in several formulas
and it would overflow.
"""


class Fock(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local fock basis."""

    def __init__(
        self,
        n_max: int | None = None,
        N: int = 1,
        n_particles: int | None = None,
        constraint: DiscreteHilbertConstraint | None = None,
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
          constraint: A custom constraint on the allowed configurations. This argument
            cannot be specified at the same time as :code:`n_particles`. The constraint
            must be a subclass of :class:`~netket.hilbert.constraint.DiscreteHilbertConstraint`

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
            if not isinstance(n_particles, int):
                raise TypeError(
                    f"n_particles must be an integer. Got {n_particles} ({type(n_particles)})"
                )
            if constraint is not None:
                raise ValueError(
                    "Cannot specify at the same time a total magnetization "
                    "constraint and a `custom_constraint."
                )

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
            constraint = SumConstraint(n_particles)
        else:
            self._n_particles = None

        if self._n_max is None:
            self._n_max = FOCK_MAX
        local_states = StaticRange(
            0,
            1,
            self._n_max + 1,
        )

        super().__init__(local_states, N, constraint=constraint)

    @property
    def n_max(self) -> int | None:
        r"""The maximum number of bosons per site, or None
        if the number is unconstrained."""
        return self._n_max

    @property
    def n_particles(self) -> int | None:
        r"""The total number of particles, or None
        if the number is unconstrained."""
        return self._n_particles

    def __pow__(self, n) -> "Fock":
        if not self.constrained:
            return Fock(self.n_max, self.size * n)

        return NotImplemented

    def _mul_sametype_(self, other: "Fock") -> "Fock":
        assert type(self) == type(other)
        if self.n_max == other.n_max:
            if not self.constrained and not other.constrained:
                return Fock(self.n_max, N=self.size + other.size)

        return NotImplemented

    def ptrace(self, sites: int | list) -> Optional["Fock"]:
        if isinstance(sites, int):
            sites = [sites]

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        if self.constrained:
            raise TypeError(
                "Cannot take the partial trace with a total particles constraint."
            )

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            return Fock(self.n_max, N=self.size - Nsites)

    def __repr__(self):
        if self.n_particles is not None:
            constraint = f", n_particles={self.n_particles}"
        elif self.constrained:
            constraint = f", {self._constraint}"
        else:
            constraint = ""

        nmax = self._n_max if self._n_max < FOCK_MAX else "FOCK_MAX"
        return f"Fock(n_max={nmax}{constraint}, N={self.size})"

    @property
    def _attrs(self):
        return (self.size, self._n_max, self.constraint)
