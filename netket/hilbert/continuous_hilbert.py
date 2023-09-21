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

    def __init__(self, geometry):
        """
        Constructs new ``Particles`` given specifications
         of the continuous space they are defined in.

        Args:
            geometry: A geometry object. Either a periodic box 'Cell' or free space 'Free'.
        """
        self._geo = geometry
        if not self._geo.extent.size == len(self._geo.pbc):
            raise ValueError(
                """`pbc` must be either a bool or a tuple indicating the periodicity of each spatial dimension."""
            )

        if np.any(np.logical_and(np.isinf(self._geo.extent), self._geo.pbc)):
            raise ValueError(
                "If you do have periodic boundary conditions in a given direction the maximum of the quantum number "
                "in that direction must be finite."
            )

        super().__init__()

    @property
    def extent(self) -> tuple[float, ...]:
        r"""Spatial extension in each spatial dimension"""
        return self._geo.extent

    @property
    def pbc(self) -> tuple[bool, ...]:
        r"""Whether or not to use periodic boundary conditions for each spatial dimension"""
        return self._geo.pbc

    @property
    def geometry(self):
        return self._geo
