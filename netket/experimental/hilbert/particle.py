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
from .continuous_hilbert import ContinuousHilbert
from netket.experimental.geometry import Cell
from netket.hilbert import TensorHilbert


class Particle(ContinuousHilbert):
    """Hilbert space of a single particle in continuous space.

    Parameters
    ----------
    geometry
        Simulation cell describing the domain.
    mass, charge, S, label
        Optional physical properties attached to the particle. They are not
        used by the Hilbert space implementation but are carried around for
        convenience.
    """

    def __init__(
        self,
        *,
        geometry: Cell,
        mass: float | None = None,
        charge: float | None = None,
        S: float | None = None,
        label: str | None = None,
    ) -> None:
        """Construct a single particle confined to ``geometry``.

        Args:
            geometry: Instance of :class:`~netket.experimental.geometry.Cell` defining
                the simulation box.
        """

        if not isinstance(geometry, Cell):
            raise TypeError(
                "`geometry` must be an instance of `netket.experimental.geometry.Cell`."
            )

        self._geometry = geometry
        self.mass = mass
        self.charge = charge
        self.S = S
        self.label = label

        super().__init__(geometry.extent)

    @property
    def size(self) -> int:
        return len(self.domain)

    @property
    def n_particles(self) -> int:
        """Number of particles represented by this Hilbert space."""
        return 1

    @property
    def geometry(self) -> Cell:
        """Geometry of the continuous space."""
        return self._geometry

    # ------------------------------------------------------------------
    # convenience utilities used by samplers

    @property
    def position_indices(self) -> tuple[int, ...]:
        """Indices corresponding to the particle coordinates."""

        return tuple(range(self.size))

    @property
    def positions_hilbert(self) -> "Particle":
        """Hilbert space describing only the particle coordinates."""

        return self

    @property
    def _attrs(self):
        return (self.geometry, self.mass, self.charge, self.S, self.label)

    def __repr__(self):
        return f"Particle(d={len(self.domain)})"

    def __pow__(self, n: int) -> "Particle | TensorHilbert":
        """Return the tensor product of ``n`` identical particles."""
        from functools import reduce

        return reduce(lambda a, b: a * b, [self] * n)
