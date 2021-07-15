from typing import Tuple

from .abstract_continuous_space_particle import AbstractParticle


class ContinuousBoson(AbstractParticle):
    r"""Hilbert space derived from AbstractParticle for
    Bosons."""

    def __init__(self, N: int, L: Tuple[float, ...], pbc: Tuple[bool, ...]):
        r"""Initializes a continuous space Hilbert space for bosons.

        Args:
        N: Number of bosonic particles.
        L: spatial extension in each spatial dimension
        pbc: Whether or not to use periodic boundary
            conditions for each spatial dimension
        """

        super().__init__(N, L, pbc)

    def __repr__(self):
        return "ContinuousBoson(N={}, d={})".format(self.n_particles, len(self.extend))
