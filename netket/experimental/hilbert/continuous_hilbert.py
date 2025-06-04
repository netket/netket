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

from netket.hilbert.abstract_hilbert import AbstractHilbert


class ContinuousHilbert(AbstractHilbert):
    """Abstract class for the Hilbert space of particles
    in continuous space.

    This class defines the common interface that
    can be used to interact with particles defined
    in continuous space.
    """

    def __init__(self, domain: tuple[float, ...]):
        """
        Constructs an Hilbert space with continuous degrees of freedom, given specifications of the space.

        This object returns an Hilbert space.

        Args:
            domain: Tuple indicating the maximum of the continuous quantum number(s) in the configurations. Each entry
                in the tuple corresponds to a different physical dimension.
                A particle in a 3D box of size L would take `(L,L,L)`. A rotor model would take e.g. `(2pi,)`.
        """
        self._extent = tuple(domain)

        super().__init__()

    @property
    def domain(self) -> tuple[float, ...]:
        r"""Domain of the continuous variable, specified for each dimension"""
        return self._extent
