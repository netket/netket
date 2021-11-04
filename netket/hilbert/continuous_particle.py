from typing import Tuple, Union, AnyStr

from .abstract_continuous_space_particle import ContinuousHilbert


class ContinuousParticle(ContinuousHilbert):
    r"""Hilbert space derived from AbstractParticle for
    Fermions."""

    def __init__(
        self, N: int, L: Tuple[float, ...], pbc: Union[bool, Tuple[bool, ...]], mode = AnyStr
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
        self._mode = mode

        if isinstance(pbc, bool):
            pbc = [pbc] * len(self._L)

        self._pbc = pbc
        if not len(self._L) == len(self._pbc):
            raise ValueError(
                """`pbc` must be either a bool or a tuple indicating the periodicity of each spatial dimension."""
            )

        super().__init__(self._L)

    @property
    def mode(self) -> AnyStr:
        return self._mode

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

    def __repr__(self):
        return "ContinuousFermion(N={}, d={}, mode={})".format(
            self.n_particles, len(self.extent), self._mode
        )