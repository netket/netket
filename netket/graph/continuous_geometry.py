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
import abc

import jax.numpy as jnp

from typing import Optional, Union, Tuple


class Geometry:
    """Documentation"""

    def __init__(
        self,
        pbc: Union[bool, tuple[bool, ...]],
        dim: Optional[int] = None,
        lattice: Optional = None,
    ):
        """
        Constructs new ``Particles`` given specifications
        of the continuous space they are defined in.

        Args:
            pbc: Tuple or bool indicating whether to use periodic boundary conditions in a given physical dimension.
                If tuple it must have the same length as domain. If bool the same value is used for all the dimensions
                defined in domain.
            dim: (Optional) The number of spatial dimensions in the Hilbert space
            lattice: (Optional) The lattice that describes the Hilbert space. For 'pbc=True' this is the basis of the
                        physical simulation cell. For 'pbc=False' the standard Euclidean basis is assumed.
            geometry: (Optional) A geometry object. Either a periodic box 'Cell' or free space 'Free'.
        """
        if dim is None and lattice is None:
            if isinstance(pbc, bool):
                raise ValueError(
                    """If 'dim' and 'lattice' is not given, the 'pbc' has to be a tuple to infer the spatial
                    dimensions"""
                )
            dim = len(pbc)

        elif lattice is None:
            self._dim = dim
        else:
            self._dim = jnp.shape(lattice)[0]

        if isinstance(pbc, bool):
            self._pbc = (pbc,) * self._dim

        if lattice is None:
            # assume cubic box if lattice is not given
            self._lat = jnp.eye(self._dim)
        else:
            self._lat = jnp.array(lattice)

        self._ilat = jnp.linalg.inv(self._lat)

        if not self._dim == len(self._pbc) == jnp.shape(self._lat)[0]:
            raise ValueError(
                """There is a mismatch between the number of spatial dimensions
                inferred by 'pbc', 'lattice' and 'dim."""
            )
        self._hash = None

    @property
    def pbc(self) -> Tuple:
        r"""Whether or not periodic boundary conditions are applied along a spatial dimension (x,y,z,...)"""
        return self._pbc

    @property
    def dim(self):
        r"""The spatial dimensions of the Hilbert space"""
        return self._dim

    @property
    def lattice(self):
        r"""The lattice used."""
        return self._lat

    @property
    def inv_lattice(self):
        r"""The reciprocal lattice to the lattice."""
        return self._ilat

    @property
    def volume(self) -> int:
        r"""The volume of the Hilbert space if defined (only defined if it is fully periodic)"""
        raise NotImplementedError

    @property
    def extent(self):
        r"""Spatial extension in each spatial dimension"""
        raise NotImplementedError

    def distance(self, x, y=None, norm=False, mode=None) -> list[list]:
        """x: (..., N, dim)
        y: None or (..., N, dim)
        norm: whether or not the norm of distances is returned as well
        mode: mode of distance computation
        """
        raise NotImplementedError

    def back_to_box(self, x):
        r"""Given coordinates in space, this function puts the coordinates back to the physical space.
        For free space this is the identity operation. For periodic boxes the fractional coordinates are
        put between 0 and 1."""
        raise NotImplementedError

    def from_lat_to_standard(self, x):
        r"""Given the coordinates of a vector x in the basis formed by the lattice, this function transforms these
        coordinates to the standard Euclidean basis.
         x = (...,N,dim)"""
        return jnp.einsum("ij,...j->...i", self._lat, x)

    def from_standard_to_lat(self, x):
        r"""Given the coordinates of a vector x in the standard Euclidean basis, this function transforms these
        coordinates to the basis formed by the lattice."""
        return jnp.einsum("ij,...j->...i", self._ilat, x)

    @property
    @abc.abstractmethod
    def _attrs(self) -> tuple:
        """
        Tuple of hashable attributes, used to compute the immutable
        hash of this Hilbert space
        """

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._attrs)

        return self._hash
