from typing import Tuple, Union

from .abstract_hilbert import AbstractHilbert


class AbstractParticle(AbstractHilbert):
    """Abstract class for the Hilbert space of particles
    in continuous space.
    This class defines the common interface that
    can be used to interact with particles defined
    in continuous space.
    """

    def __init__(
        self, N: int, L: Tuple[float, ...], pbc: Union[bool, Tuple[bool, ...]]
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
        if not len(self._L) == len(self._pbc):
            raise ValueError(
                """`pbc` must be either a bool or a tuple indicating the periodicity of each spatial dimension."""
            )

        super().__init__()

    @property
    def size(self) -> int:
        return self._N * len(self._L)

    @property
    def n_particles(self) -> int:
        r"""The number of particles"""
        return self._N

    @property
    def extend(self) -> Tuple[float, ...]:
        r"""Spatial extension in each spatial dimension"""
        return self._L

    @property
    def pbc(self) -> Tuple[bool, ...]:
        r"""Whether or not to use periodic boundary conditions for each spatial dimension"""
        return self._pbc

    @property
    def _attrs(self):
        return (
            self.size,
            self.extend,
            self.pbc,
        )
