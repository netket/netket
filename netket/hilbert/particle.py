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
from typing import Tuple, Union
import numpy as np
from .continuous_hilbert import ContinuousHilbert


class Particle(ContinuousHilbert):
    r"""Hilbert space derived from ContinuousHilbert defining N particles
    in continuous space with or without periodic boundary conditions."""

    def __init__(
        self,
        N: Union[int, Tuple[int, ...]],
        L: Tuple[float, ...],
        pbc: Union[bool, Tuple[bool, ...]],
    ):
        """
        Constructs new ``Particles`` given specifications
        of the continuous space they are defined in.

        Args:
            N: Number of particles. If int all have the same spin. If Tuple the entry indicates how many particles
                there are with a certain spin-projection.
            L: Tuple indicating the maximum of the continuous quantum number(s) in the configurations. Each entry
                in the tuple corresponds to a different physical dimension.
                If np.inf is used an infinite box is considered and `pbc=False` is mandatory (because what are PBC
                if there are no boundaries?). If a finite value is given, a minimum value of zero is assumed for the
                quantum number(s).
                A particle in a 3D box of size L would take `(L,L,L)`. A rotor model would take e.g. `(2pi,)`.
            pbc: Tuple or bool indicating whether to use periodic boundary conditions in a given physical dimension.
                If tuple it must have the same length as domain. If bool the same value is used for all the dimensions
                defined in domain.
        """
        # Assume 1D if L is a scalar
        if not hasattr(L, "__len__"):
            L = (L,)

        if not hasattr(N, "__len__"):
            N = (N,)

        if isinstance(pbc, bool):
            pbc = [pbc] * len(L)

        if np.any(np.isinf(np.array(L) * np.array(pbc))):
            raise ValueError(
                "If you do have periodic boundary conditions the size of the box (L) "
                "must be finite."
            )

        self._N = sum(N)
        self._n_per_spin = N

        super().__init__(L, pbc)

    @property
    def size(self) -> int:
        return self._N * len(self._extent)

    @property
    def n_particles(self) -> int:
        r"""The number of particles"""
        return self._N

    @property
    def n_per_spin(self) -> Tuple:
        r"""Gives the number of particles in a specific spin projection.
        The length of this tuple indicates the total spin whereas the position in the
        tuple indicates the spin projection.
        Example: (10,5,3) describes 18 particles of total spin 1 where 10 of those have spin-projection
        -1, 5 have spin-projection 0 and 3 have spin-projection 1."""
        return self._n_per_spin

    @property
    def _attrs(self):
        return (self._N, self.extent, self.pbc)

    def __repr__(self):
        return "ContinuousParticle(N={}, d={})".format(
            self.n_particles, len(self.extent)
        )
