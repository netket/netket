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
from typing import Union
from .continuous_hilbert import ContinuousHilbert


class Particle(ContinuousHilbert):
    r"""Hilbert space derived from ContinuousHilbert defining N particles
    in continuous space with or without periodic boundary conditions."""

    def __init__(
        self,
        N: Union[int, tuple[int, ...]],
        geometry,
    ):
        """
        Constructs new ``Particles`` given specifications
        of the continuous space they are defined in.

        Args:
            N: Number of particles. If int all have the same spin. If Tuple the entry indicates how many particles
                there are with a certain spin-projection..
        """
        if not hasattr(N, "__len__"):
            N = (N,)

        self._N = sum(N)
        self._n_per_spin = N
        self._geo = geometry
        super().__init__(geometry=geometry)

    @property
    def size(self) -> int:
        return self._N * self._geo.dim

    @property
    def n_particles(self) -> int:
        r"""The number of particles"""
        return self._N

    @property
    def n_per_spin(self) -> tuple:
        r"""Gives the number of particles in a specific spin projection.
        The length of this tuple indicates the total spin whereas the position in the
        tuple indicates the spin projection.
        Example: (10,5,3) describes 18 particles of total spin 1 where 10 of those have spin-projection
        -1, 5 have spin-projection 0 and 3 have spin-projection 1."""
        return self._n_per_spin

    @property
    def geometry(self):
        return self._geo

    @property
    def _attrs(self):
        return (self._N, self.pbc)

    def __repr__(self):
        return "ContinuousParticle(N={}, d={})".format(self.n_particles, self._geo.dim)
