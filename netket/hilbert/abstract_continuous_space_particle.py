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
        self._extent = domain
        if not len(self._L) == len(self._pbc):
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

    @property
    def _attrs(self):
        return (self._extent,)
