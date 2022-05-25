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

from .abstract_hilbert import AbstractHilbert


class ContinuousHilbert(AbstractHilbert):
    """Abstract class for the Hilbert space of particles
    in continuous space.
    This class defines the common interface that
    can be used to interact with particles defined
    in continuous space.
    """

    def __init__(self, domain: Tuple[float, ...], pbc: Union[bool, Tuple[bool, ...]]):
        """
        Constructs new ``Particles`` given specifications
         of the continuous space they are defined in.

        Args:
            domain: range of the continuous quantum numbers
        """
        self._extent = tuple(domain)
        self._pbc = tuple(pbc)
        if not len(self._extent) == len(self._pbc):
            raise ValueError(
                """`pbc` must be either a bool or a tuple indicating the periodicity of each spatial dimension."""
            )
        super().__init__()

    @property
    def extent(self) -> Tuple[float, ...]:
        r"""Spatial extension in each spatial dimension"""
        return self._extent

    @property
    def pbc(self) -> Tuple[bool, ...]:
        r"""Whether or not to use periodic boundary conditions for each spatial dimension"""
        return self._pbc
