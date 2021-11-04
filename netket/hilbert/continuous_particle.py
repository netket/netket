from typing import Tuple, Union, AnyStr

from .abstract_continuous_space_particle import ContinuousHilbert


class ContinuousParticle(ContinuousHilbert):
    r"""Hilbert space derived from AbstractParticle for
    Fermions."""

    def __init__(
        self,
        N: int,
        L: Tuple[float, ...],
        pbc: Union[bool, Tuple[bool, ...]],
    ):
        """
        Constructs new ``Particles`` given specifications
         of the continuous space they are defined in.
        Args:
            N: Number of particles
            L: spatial extension in each spatial dimension
            pbc: Whether or not to use periodic boundary
                conditions for each spatial dimension.
                If bool, its value will be used for all spatial
                dimensions.
        """
        self._N = N
        self._L = L

        if isinstance(pbc, bool):
            pbc = [pbc] * len(self._L)

        self._pbc = pbc

        super().__init__(self._L, self._pbc)

    @property
    def size(self) -> int:
        return self._N * len(self._L)

    @property
    def n_particles(self) -> int:
        r"""The number of particles"""
        return self._N

    def __repr__(self):
        return "ContinuousParticle(N={}, d={})".format(
            self.n_particles, len(self.extent)
        )
