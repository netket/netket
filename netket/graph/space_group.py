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
from collections.abc import Iterable, Sequence

from netket.utils import struct, deprecated_new_name, deprecated
from netket.utils.types import Array
from netket.utils.float import prune_zeros
from netket.utils.dispatch import dispatch
from netket.utils.group import (
    Element,
    Identity,
    PointGroup,
    Permutation,
    PermutationGroup,
)

from .lattice import Lattice


class Translation(Permutation):
    r"""
    Custom subclass of `Permutation` that represents a lattice permutation.
    Stores translation lattice vector and generates a sensible name from it.

    The product of two `Translation`s carries the appropriate displacement vector.
    """

    def __init__(
        self,
        permutation: Array | None = None,
        *,  # maybe change something
        displacement: Array,
        permutation_array: Array | None = None,
        inverse_permutation_array: Array | None = None,
    ):
        r"""
        Creates a `Translation` from either the array of images
        `permutation_array` or preimages `inverse_permutation_array`,
        and a displacement vector.

        Exactly one argument among `permutation_array` and
        `inverse_permutation_array` (and the deprecated argument `permutation`)
        must be specified.

        The deprecated argument `permutation` should be substituted for
        `inverse_permutation_array`.

        Note that the left action of a permutation on an array `a` is
        `a[inverse_permutation_array]`.

        Args:
            permutation: (deprecated) 1D array listing
                :math:`g^{-1}(x)` for all :math:`0\le x \le N-1`.
            displacement: displacement vector is units of lattice basis vectors
            permutation_array: 1D array listing
                :math:`g(x)` for all :math:`0\le x \le N-1`.
            inverse_permutation_array: 1D array listing
                :math:`g^{-1}(x)` for all :math:`0\le x \le N-1`.

        Returns:
            A `Translation` object that encodes the specified translation.
        """
        super().__init__(
            permutation_array=permutation_array,
            inverse_permutation_array=inverse_permutation_array,
            permutation=permutation,
        )
        self._vector = np.asarray(displacement)

    @property
    def _name(self):
        return f"Translation({self._vector.tolist()})"


@dispatch
def product(p: Translation, q: Translation):
    inverse_permutation_array = q.inverse_permutation_array[p.inverse_permutation_array]
    return Translation(
        inverse_permutation_array=inverse_permutation_array,
        displacement=p._vector + q._vector,
    )


def _ensure_iterable(x):
    """Extracts iterables given in varargs"""
    if isinstance(x[0], Iterable):
        if len(x) > 1:
            raise TypeError("Either Iterable or variable argument list expected")
        return x[0]
    else:
        return x


# This function doesn't seem to be tested
def _translations_along_axis(lattice: Lattice, axis: int) -> PermutationGroup:
    """
    The group of valid translations along an axis as a `PermutationGroup`
    acting on the sites of `lattice.`
    """
    if lattice._pbc[axis]:
        trans_list = [Identity()]
        # note that we need the preimages in the permutation
        trans_perm = lattice.id_from_position(
            lattice.positions - lattice.basis_vectors[axis]
        )
        vector = np.zeros(lattice.ndim, dtype=int)
        vector[axis] = 1
        trans_by_one = Translation(
            inverse_permutation_array=trans_perm, displacement=vector
        )

        for _ in range(1, lattice.extent[axis]):
            trans_list.append(trans_list[-1] @ trans_by_one)

        return PermutationGroup(trans_list, degree=lattice.n_nodes)
    else:
        return PermutationGroup([Identity()], degree=lattice.n_nodes)


# This function doesn't seem to be tested
def _pg_to_permutation(lattice: Lattice, point_group: PointGroup) -> PermutationGroup:
    """
    The permutation action of `point_group` on the sites of `lattice`.
    """
    perms: list[Element] = []
    for p in point_group:
        if isinstance(p, Identity):
            perms.append(Identity())
        else:
            # note that we need the preimages in the permutation
            perm = lattice.id_from_position(p.preimage(lattice.positions))
            perms.append(Permutation(inverse_permutation_array=perm, name=str(p)))
    return PermutationGroup(perms, degree=lattice.n_nodes)


@struct.dataclass
class TranslationGroup(PermutationGroup):
    """
    Class to handle translation symmetries of a `Lattice`. Corresponds to a representation of the translation group
    on the given lattice as a permutation group of `N_sites` variables.

    Can be used as a `PermutationGroup` representing the translations,
    but the product table is computed much more efficiently than a generic
    `PermutationGroup`.
    """

    lattice: Lattice
    """The lattice whose translation group is represented."""
    axes: tuple[int]
    """Axes translations along which are represented by the group."""

    def __repr__(self):
        return type(self).__name__ + f"(lattice:\n{self.lattice}\naxes:{self.axes})"

    def __pre_init__(
        self, lattice: Lattice, axes: int | tuple[int] | None = None
    ) -> tuple[tuple, dict]:
        if axes is None:
            axes = tuple(range(lattice.ndim))
        elif isinstance(axes, int):
            axes = [axes]
        else:
            assert all(x < lattice.ndim for x in axes)
            assert len(set(axes)) == len(axes)

        # compute translation group by axis and overall
        translation_by_axis = [_translations_along_axis(lattice, i) for i in axes]
        translation_group = reduce(PermutationGroup.__matmul__, translation_by_axis)

        return (), dict(
            lattice=lattice,
            axes=tuple(axes),
            elems=translation_group.elems,
            degree=lattice.n_nodes,
        )

    def __hash__(self):
        return super().__hash__()

    @property
    def group_shape(self) -> Array:
        shape = [l if p else 1 for (l, p) in zip(self.lattice.extent, self.lattice.pbc)]
        return np.asarray(shape)

    @struct.property_cached
    def inverse(self) -> Array:
        ix = np.ix_(*[-np.arange(x) % x for x in self.group_shape])
        inv = np.arange(len(self)).reshape(self.group_shape)[ix]
        return np.ravel(inv)

    @struct.property_cached
    def product_table(self) -> Array:
        # product table of each 1d component
        ix = [(np.arange(x) - np.arange(x)[:, None]) % x for x in self.group_shape]
        ix = np.ix_(*[np.ravel(x) for x in ix])
        # product table with n_axes axes
        # each axis stands for row and column direction along one axis
        pt = np.arange(len(self)).reshape(self.group_shape)[ix]
        shape = [x for x in self.group_shape for _ in range(2)]
        # separate row and column directions
        pt = pt.reshape(shape)
        # bring all row and all column directions together
        pt = pt.transpose(list(range(0, len(shape), 2)) + list(range(1, len(shape), 2)))

        return pt.reshape(len(self), len(self))


@struct.dataclass
class SpaceGroup(PermutationGroup):
    """
    Class to handle the space group symmetries of `Lattice`.

    Can be used as a `PermutationGroup` representing the action of a
    space group on a `Lattice`. The space group is generated as the
    semidirect product of the translation group of the `Lattice` and
    a geometrical point group given as a constructor argument.

    Also generates `PermutationGroup` representations of
    * the supplied point group,
    * its rotational subgroup (i.e. point group symmetries with determinant +1)
    * the translation group of the `Lattice`

    Also generates space group irreps for symmetrising wave functions.
    """

    lattice: Lattice
    """The lattice underlying the space group."""
    _point_group: PointGroup
    """The geometric point group underlying the space group."""
    point_group: PermutationGroup
    """The point group as a `PermutationGroup` acting on the sites of `self.lattice`.

    Group elements are listed in the order they appear in `self._point_group`.
    Computed from `_point_group` upon construction, must not be changed after."""
    full_translation_group: PermutationGroup

    def __pre_init__(
        self, lattice: Lattice, point_group: PointGroup
    ) -> tuple[tuple, dict]:
        """
        Constructs the Space Group Builder used to concretize a point group
        which knows nothing about how many sites there are in a lattice,
        into a Permutation Group which can be used to perform calculations.

        From the point of view of group theory, you can think of this as
        taking the point group and returning a representation on the computational
        basis defined by the lattice.

        Args:
            lattice: The lattice for which to represent the point group as
                a permutation group
            point_group: The point group to be represented
        """

        if not isinstance(lattice, Lattice):
            raise TypeError(
                "Expected an instance of `Lattice` as argument `lattice`, "
                f"got {type(lattice)}."
            )

        if not isinstance(point_group, PointGroup):
            msg = (
                "Expected an instance of `PointGroup` as argument `point_group`, "
                f"got type {type(point_group)}."
            )
            if isinstance(point_group, Element):
                msg += (
                    "\n\n`point_group` appears to be a single symmetry. "
                    "It should be wrapped into a `PointGroup` object."
                )
            raise TypeError(msg)

        # compute point group permutations
        point_group = point_group.replace(unit_cell=lattice.basis_vectors)
        pg_as_perm = _pg_to_permutation(lattice, point_group)

        # compute translation group by axis and overall
        translation_group = lattice.full_translation_group

        # compute space group elements
        space_group = translation_group @ pg_as_perm
        elems = space_group.elems

        return (), dict(
            lattice=lattice,
            _point_group=point_group,
            point_group=pg_as_perm,
            full_translation_group=translation_group,
            elems=elems,
            degree=lattice.n_nodes,
        )

    def __hash__(self):
        return super().__hash__()

    def __repr__(self):
        return (
            type(self).__name__
            + f"(lattice:\n{self.lattice}\npoint_group:\n{self._point_group})"
        )

    @property
    @deprecated_new_name("_point_group", reason="Consistency")
    def point_group_(self) -> PointGroup:
        """
        Deprecated: Returns the internally stored point group as a point group,
        instead of the one stored as a permutation group.
        """

        return self._point_group

    # TODO describe ordering of group elements here and later in docstring

    @struct.property_cached
    def rotation_group(self) -> PermutationGroup:
        """The group of rotations (i.e. point group symmetries with determinant +1)
        as a `PermutationGroup` acting on the sites of `self.lattice`.

        Group elements are listed in the order they appear in `self._point_group`."""
        return _pg_to_permutation(self.lattice, self._point_group.rotation_group())

    def translation_group(
        self, axes: int | Sequence[int] | None = None
    ) -> TranslationGroup:
        """
        The group of valid translations of `self.lattice` as a `PermutationGroup`
        acting on the sites of the same.
        """
        if axes is None:
            return self.full_translation_group
        else:
            return TranslationGroup(self.lattice, axes=axes)

    @property
    @deprecated(
        reason="This `SpaceGroup` object can be used directly as a permutation group"
    )
    def space_group(self) -> "SpaceGroup":
        """
        The space group generated by `self.point_group` and `self.translation_group`.
        """
        return self

    @struct.property_cached
    def product_table(self) -> Array:
        # compute first n_PG rows of product table like in PermutationGroup
        perms = self.to_array()
        inverse = perms[self.inverse].squeeze()
        n_symm = len(perms)
        n_PG = len(self._point_group)
        n_TG = len(self.full_translation_group)
        lookup = np.unique(np.column_stack((perms, np.arange(len(self)))), axis=0)

        PG_rows = np.zeros([n_PG, n_symm], dtype=int)
        for i, g_inv in enumerate(inverse[:n_PG]):
            row_perms = perms[:, g_inv]
            row_perms = np.unique(
                np.column_stack((row_perms, np.arange(len(self)))), axis=0
            )
            # row_perms should be a permutation of perms, so identical after sorting
            if np.any(row_perms[:, :-1] != lookup[:, :-1]):
                raise RuntimeError(
                    "PermutationGroup is not closed under multiplication"
                )
            # match elements in row_perms to group indices
            PG_rows[i, row_perms[:, -1]] = lookup[:, -1]

        # PG_rows contains pg^-1 th ph - split three terms into three dimensions
        PG_rows = PG_rows.reshape(n_PG, n_TG, n_PG)
        # the full product table is of the form pg^-1 tg^-1 th ph
        # the middle two terms are the product table of the TG
        product_table = PG_rows[:, self.full_translation_group.product_table, :]
        # reshuffle into output shape
        product_table = product_table.transpose(1, 0, 2, 3)

        return product_table.reshape(n_symm, n_symm)

    def _little_group_index(self, k: Array) -> Array:
        """
        Returns the indices of the elements of the little group corresponding to
        wave vector `k`.
        """
        # calculate k' = p(k) for all p in the point group
        big_star = np.tensordot(self._point_group.matrices(), k, axes=1)
        big_star = self.lattice.to_reciprocal_lattice(big_star) % self.lattice.extent
        # should test for pbc before taking the modulus, but the only valid wave
        # vector for non-pbc axes is 0 and 0 % anything == 0

        # assumes point_group_[0] is the identity
        is_in_little_group = np.all(big_star == big_star[0], axis=1)
        return np.arange(len(self._point_group))[is_in_little_group]

    def little_group(self, *k: Array) -> PointGroup:
        """
        Returns the little co-group corresponding to wave vector *k*.
        This is the subgroup of the point group that leaves *k* invariant.

        Arguments:
            k: the wave vector in Cartesian axes

        Returns:
            the little co-group as a `PointGroup`
        """
        k = _ensure_iterable(k)
        return PointGroup(
            [self._point_group[i] for i in self._little_group_index(k)],
            ndim=self._point_group.ndim,
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
        CT_full = np.zeros((CT.shape[0], len(self._point_group)), dtype=CT.dtype)
        CT_full[:, idx] = CT
        return CT_full / idx.size if divide else CT_full

    def space_group_irreps(self, *k: Array) -> Array:
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
        k = _ensure_iterable(k)
        # Wave vectors
        big_star_Cart = np.tensordot(self._point_group.matrices(), k, axes=1)
        big_star = self.lattice.to_reciprocal_lattice(big_star_Cart) * (
            2 * pi / self.lattice.extent
        )
        # Little-group-irrep factors
        # Conjugacy_table[g,p] lists p^{-1}gp, so point_group_factors[i,:,p]
        #     of irrep #i for the little group of p(k) is the equivalent
        # Phase factor for non-symmorphic symmetries is exp(-i w_g . p(k))
        point_group_factors = self._little_group_irreps(k, divide=True)[
            :, self._point_group.conjugacy_table
        ] * np.exp(
            -1j
            * np.tensordot(
                self._point_group.translations(), big_star_Cart, axes=(-1, -1)
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
                + [len(self._point_group)]
            )
            trans_factors.append(factors.reshape(shape))
        trans_factors = reduce(np.multiply, trans_factors).reshape(
            -1, len(self._point_group)
        )

        # Multiply the factors together and sum over the "p" PGSymmetry axis
        # Translations are more major than point group operations
        result = np.einsum(
            "igp, tp -> itg", point_group_factors, trans_factors
        ).reshape(point_group_factors.shape[0], -1)
        return prune_zeros(result)

    def one_arm_irreps(self, *k: Array) -> Array:
        """
        Returns the portion of the character table of the full space group corresponding
        to the star of the wave vector *k*, projected onto *k* itself.

        Arguments:
            k: the wave vector in Cartesian axes

        Returns:
            An array `CT` listing the projected characters for a number of irreps of
            the space group.
            `CT[i]` for each `i` gives a distinct irrep, each corresponding to
            `self.little_group(k).character_table[i].
            `CT[i,j]` gives the character of `self.space_group[j]` in the same.
        """
        # Convert k to reciprocal lattice vectors
        k = _ensure_iterable(k)
        # Little-group irrep factors
        # Phase factor for non-symmorphic symmetries is exp(-i w_g . p(k))
        point_group_factors = self._little_group_irreps(k) * np.exp(
            -1j * (self._point_group.translations() @ k)
        )
        # Translational factors
        trans_factors = []
        for axis in range(self.lattice.ndim):
            n_trans = self.lattice.extent[axis] if self.lattice.pbc[axis] else 1
            factors = np.exp(-1j * k[axis] * np.arange(n_trans))
            shape = [1] * axis + [n_trans] + [1] * (self.lattice.ndim - 1 - axis)
            trans_factors.append(factors.reshape(shape))
        trans_factors = reduce(np.multiply, trans_factors).ravel()

        # Multiply the factors together
        # Translations are more major than point group operations
        result = np.einsum("ig, t -> itg", point_group_factors, trans_factors).reshape(
            point_group_factors.shape[0], -1
        )
        return prune_zeros(result)
