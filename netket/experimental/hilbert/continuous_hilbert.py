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

from .abstract_hilbert import AbstractHilbert

import numpy as np


class ContinuousHilbert(AbstractHilbert):
    """Abstract class for the Hilbert space of particles
    in continuous space.

    This class defines the common interface that
    can be used to interact with particles defined
    in continuous space.
    """

    def __init__(self, domain: tuple[float, ...], pbc: bool | tuple[bool, ...]):
        """
        Constructs new ``Particles`` given specifications of the continuous
        space they are defined in.

        This object returns an Hilber space.

        Args:
            domain: Tuple indicating the maximum of the continuous quantum number(s) in the configurations. Each entry
                in the tuple corresponds to a different physical dimension.
                If np.inf is used an infinite box is considered and `pbc=False` is mandatory (because what are PBC
                if there are no boundaries?). If a finite value is given, a minimum value of zero is assumed for the
                quantum number(s).
                A particle in a 3D box of size L would take `(L,L,L)`. A rotor model would take e.g. `(2pi,)`.
            pbc: Tuple or bool indicating whether to use periodic boundary conditions in a given physical dimension.
                If tuple it must have the same length as domain. If bool the same value is used for all the dimensions
                defined in domain.
        """
        self._extent = tuple(domain)
        self._pbc = tuple(pbc)
        if not len(self._extent) == len(self._pbc):
            raise ValueError(
                """`pbc` must be either a bool or a tuple indicating the periodicity of each spatial dimension."""
            )

        if np.any(np.logical_and(np.isinf(self._extent), pbc)):
            raise ValueError(
                "If you do have periodic boundary conditions in a given direction the maximum of the quantum number "
                "in that direction must be finite."
            )

        super().__init__()

    @property
    def extent(self) -> tuple[float, ...]:
        r"""Spatial extension in each spatial dimension"""
        return self._extent

    @property
    def pbc(self) -> tuple[bool, ...]:
        r"""Whether or not to use periodic boundary conditions for each spatial dimension"""
        return self._pbc
