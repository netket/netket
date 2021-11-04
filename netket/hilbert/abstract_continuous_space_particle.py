from typing import Tuple

from .abstract_hilbert import AbstractHilbert


class ContinuousHilbert(AbstractHilbert):
    """Abstract class for the Hilbert space of particles
    in continuous space.
    This class defines the common interface that
    can be used to interact with particles defined
    in continuous space.
    """

    def __init__(self, domain: Tuple[float, ...]):
        """
        Constructs new ``Particles`` given specifications
         of the continuous space they are defined in.

        Args:
            domain: range of the continuous quantum numbers
        """
        self._extent = domain

        super().__init__()

    @property
    def extent(self) -> Tuple[float, ...]:
        r"""Spatial extension in each spatial dimension"""
        return self._extent

    @property
    def _attrs(self):
        return (self._extent,)
