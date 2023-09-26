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
from typing import Optional, Union
import jax.numpy as jnp
import numpy as np
from .continuous_hilbert import ContinuousHilbert


class Particle(ContinuousHilbert):
    r"""Hilbert space derived from ContinuousHilbert defining N particles
    in continuous space with or without periodic boundary conditions."""

    def __init__(
        self,
        N: Union[int, tuple[int, ...]],
        L: Optional[tuple[float, ...]] = None,
        pbc: Optional[bool] = None,
        *,
        D: Optional[int] = None,
        geometry: Optional = None,
    ):
        """
        Constructs new ``Particles`` given specifications
        of the continuous space they are defined in.

        Args:
            N: Number of particles. If int all have the same spin. If Tuple the entry indicates how many particles
                there are with a certain spin-projection.
            L: Tuple indicating the maximum of the continuous quantum number(s) in the configurations. Each entry
                in the tuple corresponds to a different physical dimension.
                If `np.inf` is used an infinite box is considered and `pbc=False` is mandatory (because what are PBC
                if there are no boundaries?). If a finite value is given, a minimum value of zero is assumed for the
                quantum number(s).
                A particle in a 3D box of size L would take `(L,L,L)`. A rotor model would take e.g. `(2pi,)`.
            pbc: Tuple of bools or bool indicating whether to use periodic boundary conditions in a given physical
                 dimension. Mixed boundary conditions are not supported at the moment i.e. if tuple it must have the
                 same length as L and the same entry in all directions.
            D: (Optional) Number of dimensions. Can be specified instead of `L` and `pbc` in order to construct a
                `Particle` in a $D-$ dimensional infinite box. Equivalent to `Particle(N, L=(np.inf,) * D, pbc=False)`.
            geometry: (Optional) Geometry object describing the physical space. Can be specified instead of `L` and
                      `pbc` as each geometry object contains this information.
                      Available options are free particles
                      'Free' equivalent to `Particle(N, L=(np.inf,) * geometry.dim, pbc=False) or particles in a box
                      'Cell' equivalent to `Particle(N, L=geometry.basis, pbc=True).
                       If 'D' is specified, giving a geometry errors because 'D' implies free space.
        """
        if D is None and L is None and geometry is None:
            raise ValueError("Must specify at least `L` or `D` or a geometry.")

        elif D is not None:
            if L is not None:
                raise TypeError(
                    "Cannot specify at the same time `D` and `L`. If you want to use "
                    "an infinite box, just specify D, otherwise specify L."
                )
            if geometry is not None:
                raise TypeError(
                    "Cannot specify at the same time 'D' and 'geometry'. If you want to use an infinite box, specify D"
                    "or input a 'Free' geometry object. If you want to use a periodic box, specify a 'Cell' geometry"
                    "object."
                )
            from netket.experimental.geometry.Free import Free

            geometry = Free(basis=jnp.eye(D))

        elif L is not None:
            if geometry is not None:
                raise TypeError(
                    "Cannot specify at the same time 'L' and 'geometry'. Either specify 'L' and 'pbc' or provide a "
                    "geometry."
                )
            # Assume 1D if L is a scalar
            if not hasattr(L, "__len__"):
                L = (L,)
            if pbc is None:
                if np.all(np.isinf(L)):
                    pbc = False
                else:
                    raise ValueError("`pbc` must be specified if `L` is finite.")
            if np.any(np.logical_and(np.isinf(L), pbc)):
                raise ValueError(
                    "Cannot combine periodic boundary conditions and infinite size along the same dimension."
                )

            if isinstance(pbc, bool):
                pbc = (pbc,) * len(L)
            if all(map(lambda x: x is True, pbc)):
                from netket.experimental.geometry.Cell import Cell

                geometry = Cell(basis=jnp.array(L) * jnp.eye(len(L)))
            elif all(map(lambda x: x is False, pbc)):
                from netket.experimental.geometry.Free import Free

                geometry = Free(basis=jnp.array(L) * jnp.eye(len(L)))
            else:
                raise ValueError("Mixed boundary conditions are not supported.")

        if not hasattr(N, "__len__"):
            N = (N,)

        self._N = sum(N)
        self._n_per_spin = N

        super().__init__(geometry)

    @property
    def size(self) -> int:
        return self._N * self.geometry.dim

    @property
    def n_particles(self) -> int:
        r"""The number of particles"""
        return self._N

    @property
    def n_per_spin(self) -> tuple:
        r"""Gives the number of particles in a specific spin projection.
        The length of this tuple indicates the total spin whereas the position in the
        tuple indicates the spin projection.
        Example: (10,5,3) describes 18 particles of total spin 1 where 10 of those have spin-projection
        -1, 5 have spin-projection 0 and 3 have spin-projection 1."""
        return self._n_per_spin

    @property
    def _attrs(self):
        return (self._N, self.geometry)

    def __repr__(self):
        return "ContinuousParticle(N={}, d={})".format(self.n_particles, self.geometry)
