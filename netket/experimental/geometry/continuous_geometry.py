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

from typing import Optional


class AbstractGeometry(abc.ABC):
    """Documentation"""

    def __init__(
        self,
        dim: Optional[int] = None,
        basis: Optional = None,
    ):
        """
        Construct a geometry in continuous space, given the dimension or a basis. If only the dimension is given
        the standard basis is assumed, e.g. (1,0), (0,1) for 2D space.

        Args:
            dim: (Optional) The number of spatial dimensions of the physical space. If None, a basis has to be
                            specified. If int and basis is None, the standard basis is assumed.
            basis: (Optional) A basis for the physical space. If basis is None, the standard basis is assumed.
        """
        if dim is None and basis is None:
            raise ValueError(
                """Specify either the dimension of the geometry or provide a basis for it."""
            )

        if basis is None:
            basis = jnp.eye(dim)

        self._basis = basis
        self._dim = self._basis.shape[0]

        self._hash = None

    @property
    def dim(self):
        r"""The number of spatial dimensions of the geometry."""
        return self._dim

    @property
    def basis(self):
        r"""The basis spanning the geometry."""
        return self._basis

    @property
    def inv_basis(self):
        r"""The reciprocal space of the given geometry."""
        return jnp.linalg.inv(self.basis)

    @abc.abstractmethod
    def distance(self, x, y=None, norm=False, mode=None, tri=False) -> list[list]:
        r"""Distance function for the given geometry. For free space this is just the Euclidean distance. For
        periodic geometries, this could be Minimum image convention or smooth periodic distance.
           Args:
               x: Array (..., N, dim), batch of samples containing N positions with the same dimension as the geometry.
               y: (Optional) Array (..., M, dim), batch of samples containing M positions with the same dimension
                                as the geometry. y = None means distance(x,x) with a safe norm.
               norm: bool, indicates whether to return only distance vectors or also their norm.
               mode: str, indicates which distance is computed. Options: 'Euclidean' is the standard distance.
                            'MIC' denotes the minimum image distance. 'Periodic' denotes a smooth periodic distance.
               tri: bool, indicates whether the whole distance matrix is returned or its upper triangular part.
           Returns: Array (..., N, M, dim), Distance matrix where the ij-th entry contains the distance between
                                            the particle in position x[...,i,:] and in position y[...,j,:].
        """

    @abc.abstractmethod
    def add(self, x, y):
        r"""Given two batches of position vectors in the geometry, this function defines how these vectors are added
        together. For free space, this is the usual vector addition. FOr periodic space, this includes folding particle
        positions back to the geometry defined."""

    @abc.abstractmethod
    def random_init(self, shape):
        r"""This defines an initial configuration in the geometry from which the random_state in Hilbert is constructed.
        For free space this could just return zeros. For periodic space one could initialize on a lattice.
        N: int, number of particles."""

    def from_basis_to_standard(self, x):
        r"""Given a batch of position vectors in the geometry, expressed in the basis of the geometry (self.basis),
        this function returns the position vectors expressed in the standard basis."""
        return jnp.einsum("ij,...j->...i", self.basis, x)

    def from_standard_to_basis(self, x):
        r"""Given a batch of position vectors in the geometry, expressed in the standard basis,
        this function returns the position vectors expressed in the geometry basis (self.basis).
        """
        return jnp.einsum("ij,...j->...i", self.inv_basis, x)

    @property
    @abc.abstractmethod
    def _attrs(self) -> tuple:
        """
        Tuple of hashable attributes, used to compute the immutable
        hash of this Hilbert space
        """

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return self._attrs == other._attrs
        return False

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._attrs)

        return self._hash
