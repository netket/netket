from typing import Tuple

from .abstract_hilbert import AbstractHilbert

class Particles(AbstractHilbert):
    """Class for continuous Hilbert space.

    This class definese the common interface that can be used to
    interact with hilbert spaces in continuum.
    """

    def __init__(self, N: int, L: Tuple[float, ...], pbc: Tuple[bool, ...], ptype: str):
        """
        Initializes a continuous Hilbert space.

        :param N: Number of particles
        :param L: spatial extension in each spatial dimension
        :param pbc: Whether or not to use periodic boundary conditions for each spatial dimension
        :param type: Type of particles (bosonic,..)
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
        r"""The number number of degrees of freedom in the
                Hilbert space."""
        return self._N * len(self._L)

    @property
    def N(self) -> int:
        r"""The number of particles
        """
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
        return self._type

    @property
    def _attrs(self):
        return (
            self.size,
            self.L,
            self.pbc,
            self.type,
        )