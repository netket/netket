from typing import Tuple

from .abstract_hilbert import AbstractHilbert

class Particles(AbstractHilbert):
    """Class for the Hilbert space of particles
    in continuous space.
    This class defines the common interface that
    can be used to interact with particles defined
    in continuous space.
    """

    def __init__(self, N: int, L: Tuple[float, ...], pbc: Tuple[bool, ...], ptype: str):
        """
        Constructs new ``Particles`` given specifications
         of the continuous space they are defined in.

        Args:
            N: Number of particles
            L: spatial extension in each spatial dimension
            pbc: Whether or not to use periodic boundary
                conditions for each spatial dimension
            ptype: Type of particles (bosonic,..)
        """
        self._N = N
        self._L = L
        self._pbc = pbc
        self._ptype = ptype

        if not len(self._L) == len(self._pbc):
            raise AssertionError(
                """You need to define boundary conditions for each spatial dimension."""
            )

        super().__init__()

    @property
    def size(self) -> int:

        return self._N * len(self._L)

    @property
    def N(self) -> int:
        r"""The number of particles"""
        return self._N

    @property
    def L(self) -> Tuple[float, ...]:
        r"""Spatial extension in each spatial dimension"""
        return self._L

    @property
    def pbc(self) -> Tuple[bool, ...]:
        r"""Whether or not to use periodic boundary conditions for each spatial dimension"""
        return self._pbc

    @property
    def ptype(self):
        r"""The type of particles under consideration"""
        return self._ptype

    @property
    def _attrs(self):
        return (
            self.size,
            self.L,
            self.pbc,
            self.ptype,
        )
