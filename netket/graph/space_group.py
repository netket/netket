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

# Ignore false-positives for redefined `product` functions:
# pylint: disable=function-redefined

import numpy as np
from functools import reduce
from math import pi
from typing import Optional, Sequence

from .lattice import Lattice

from netket.utils import struct
from netket.utils.types import Array, Union
from netket.utils.float import prune_zeros
from netket.utils.dispatch import dispatch

from netket.utils.group import (
    Identity,
    PointGroup,
    Permutation,
    PermutationGroup,
)


class Translation(Permutation):
    r"""
    Custom subclass of `Permutation` that represents a lattice permutation.
    Stores translation lattice vector and generates a sensible name from it.

    The product of two `Translation`s carries the appropriate displacement vector.
    """

    def __init__(self, permutation: Array, displacement: Array):
        r"""
        Creates a `Translation` from a permutation array and a displacement vector

        Arguments:
            permutation: a 1D array listing :math:`g^{-1}(x)` for all
                :math:`0\le x < N` (i.e., `V[permutation]` permutes the
                elements of `V` as desired)
            displacement: displacement vector is units of lattice basis vectors

        Returns:
            a `Translation` object encoding the same information
        """
        super().__init__(permutation)
        self._vector = np.asarray(displacement)

    @property
    def _name(self):
        return f"Translation({self._vector.tolist()})"


@dispatch
def product(p: Translation, q: Translation):
    return Translation(p(np.asarray(q)), p._vector + q._vector)


@struct.dataclass
class SpaceGroupBuilder:
    """
    Class to handle the space group symmetries of `Lattice`.

    Constructs `PermutationGroup`s that represent the action on a `Lattice` of
    * a geometrical point group given as a constructor argument,
    * its rotational subgroup (i.e. point group symmetries with determinant +1)
    * the translation group of the same lattice
    * and the space group that is generated as the semidirect product of
      the supplied point group and the translation group.

    Also generates space group irreps for symmetrising wave functions.
    """

    lattice: Lattice
    point_group_: PointGroup

    def __post_init__(self):
        object.__setattr__(
            self,
            "point_group_",
            self.point_group_.replace(unit_cell=self.lattice.basis_vectors),
        )

    # TODO describe ordering of group elements here and later in docstring
    @struct.property_cached
    def point_group(self) -> PermutationGroup:
        """
        The point group as a `PermutationGroup` acting on the sites of `self.lattice`.
        """
        perms = []
        for p in self.point_group_:
            if isinstance(p, Identity):
                perms.append(Identity())
            else:
                # note that we need the preimages in the permutation
                perm = self.lattice.id_from_position(p.preimage(self.lattice.positions))
                perms.append(Permutation(perm, name=str(p)))
        return PermutationGroup(perms, degree=self.lattice.n_nodes)

    @struct.property_cached
    def rotation_group(self) -> PermutationGroup:
        """The group of rotations (i.e. point group symmetries with determinant +1)
        as a `PermutationGroup` acting on the sites of `self.lattice`."""
        perms = []
        for p in self.point_group_.rotation_group():
            if isinstance(p, Identity):
                perms.append(Identity())
            else:
                # note that we need the preimages in the permutation
                perm = self.lattice.id_from_position(p.preimage(self.lattice.positions))
                perms.append(Permutation(perm, name=str(p)))
        return PermutationGroup(perms, degree=self.lattice.n_nodes)

    def _translations_along_axis(self, axis: int) -> PermutationGroup:
        """
        The group of valid translations along an axis as a `PermutationGroup`
        acting on the sites of `self.lattice.`
        """
        if self.lattice._pbc[axis]:
            trans_list = [Identity()]
            # note that we need the preimages in the permutation
            trans_perm = self.lattice.id_from_position(
                self.lattice.positions - self.lattice.basis_vectors[axis]
            )
            vector = np.zeros(self.lattice.ndim, dtype=int)
            vector[axis] = 1
            trans_by_one = Translation(trans_perm, vector)

            for _ in range(1, self.lattice.extent[axis]):
                trans_list.append(trans_list[-1] @ trans_by_one)

            return PermutationGroup(trans_list, degree=self.lattice.n_nodes)
        else:
            return PermutationGroup([Identity()], degree=self.lattice.n_nodes)

    @struct.property_cached
    def _full_translation_group(self) -> PermutationGroup:
        """
        The group of valid translations of `self.lattice` as a `PermutationGroup`
        acting on the sites of the same.
        """
        return reduce(
            PermutationGroup.__matmul__,
            [self._translations_along_axis(i) for i in range(self.lattice.ndim)],
        )

    def translation_group(
        self, axes: Optional[Union[int, Sequence[int]]] = None
    ) -> PermutationGroup:
        """
        The group of valid translations of `self.lattice` as a `PermutationGroup`
        acting on the sites of the same.
        """
        if axes is None:
            return self._full_translation_group
        elif isinstance(axes, int):
            return self._translations_along_axis(axes)
        else:
            return reduce(
                PermutationGroup.__matmul__,
                [self._translations_along_axis(i) for i in axes],
            )

    @struct.property_cached
    def space_group(self) -> PermutationGroup:
        """
        The space group generated by `self.point_group` and `self.translation_group`.
        """
        return self._full_translation_group @ self.point_group

    def _little_group_index(self, k: Array) -> Array:
        """
        Returns the indices of the elements of the little group corresponding to
        wave vector `k`.
        """
        # calculate k' = p(k) for all p in the point group
        big_star = np.tensordot(self.point_group_.matrices(), k, axes=1)
        big_star = self.lattice.to_reciprocal_lattice(big_star) % self.lattice.extent
        # should test for pbc before taking the modulus, but the only valid wave
        # vector for non-pbc axes is 0 and 0 % anything == 0

        # assumes point_group_[0] is the identity
        is_in_little_group = np.all(big_star == big_star[0], axis=1)
        return np.arange(len(self.point_group_))[is_in_little_group]

    def little_group(self, k: Array) -> PointGroup:
        """
        Returns the little co-group corresponding to wave vector *k*.
        This is the subgroup of the point group that leaves *k* invariant.

        Arguments:
            k: the wave vector in Cartesian axes

        Returns:
            the little co-group as a `PointGroup`
        """
        return PointGroup(
            [self.point_group_[i] for i in self._little_group_index(k)],
            ndim=self.point_group_.ndim,
            unit_cell=self.lattice.basis_vectors,
        )

    def _little_group_irreps(self, k: Array, divide: bool = False) -> Array:
        """
        Returns the character table of the little group embedded in the full point
        group. Symmetries outside the little group get 0.
        If `divide` is `True`, the result gets divided by the size of the little group.
        This is convenient when calculating space group irreps.
        """
        idx = self._little_group_index(k)
        CT = self.little_group(k).character_table()
        CT_full = np.zeros((CT.shape[0], len(self.point_group_)))
        CT_full[:, idx] = CT
        return CT_full / idx.size if divide else CT_full

    def space_group_irreps(self, k: Array) -> Array:
        """
        Returns the portion of the character table of the full space group corresponding
        to the star of the wave vector *k*.

        Arguments:
            k: the wave vector in Cartesian axes

        Returns:
            An array `CT` listing the characters for a number of irreps of the
            space group.
            `CT[i]` for each `i` gives a distinct irrep, each corresponding to
            `self.little_group(k).character_table[i].
            `CT[i,j]` gives the character of `self.space_group[j]` in the same.
        """
        # Wave vectors
        big_star_Cart = np.tensordot(self.point_group_.matrices(), k, axes=1)
        big_star = self.lattice.to_reciprocal_lattice(big_star_Cart) * (
            2 * pi / self.lattice.extent
        )
        # Little-group-irrep factors
        # Conjugacy_table[g,p] lists p^{-1}gp, so point_group_factors[i,:,p]
        #     of irrep #i for the little group of p(k) is the equivalent
        # Phase factor for non-symmorphic symmetries is exp(-i w_g . p(k))
        point_group_factors = self._little_group_irreps(k, divide=True)[
            :, self.point_group_.conjugacy_table
        ] * np.exp(
            -1j
            * np.tensordot(
                self.point_group_.translations(), big_star_Cart, axes=(-1, -1)
            )
        )
        # Translational factors
        trans_factors = []
        for axis in range(self.lattice.ndim):
            n_trans = self.lattice.extent[axis] if self.lattice.pbc[axis] else 1
            factors = np.exp(-1j * np.outer(np.arange(n_trans), big_star[:, axis]))
            shape = (
                [1] * axis
                + [n_trans]
                + [1] * (self.lattice.ndim - 1 - axis)
                + [len(self.point_group_)]
            )
            trans_factors.append(factors.reshape(shape))
        trans_factors = reduce(np.multiply, trans_factors).reshape(
            -1, len(self.point_group_)
        )

        # Multiply the factors together and sum over the "p" PGSymmetry axis
        # Translations are more major than point group operations
        result = np.einsum(
            "igp, tp -> itg", point_group_factors, trans_factors
        ).reshape(point_group_factors.shape[0], -1)
        return prune_zeros(result)
