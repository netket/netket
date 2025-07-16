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
from collections.abc import Iterable, Sequence
from warnings import warn

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
    Class to handle translation symmetries of a :class:`~netket.graph.Lattice`.

    Corresponds to a representation of the translation group
    on the given lattice as a permutation group of :code:`N_sites` variables.

    Can be used as a :class:`~netket.utils.group.PermutationGroup` representing
    the translations, but the product table is computed much more efficiently
    than in a generic :class:`~netket.utils.group.PermutationGroup`.
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

        if len(set(axes)) != len(axes):
            raise ValueError(
                f"Axes must be unique integers, they cannot repeat (got `{axes}`)."
            )
        if not all(0 <= x < lattice.ndim for x in axes):
            raise ValueError(
                f"Axes must be integers in the range [0, {lattice.ndim}) (got `{axes}`)."
            )

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

    @struct.property_cached
    def group_shape(self) -> Array:
        """
        Tuple of the number of translations represented by the group along
        each lattice direction.

        :code:`self.group_shape[i]` is :code:`self.lattice.extent[i]`
        if both :code:`i in self.axes` and :code:`self.lattice.pbc[i] is True`,
        otherwise 1.
        """
        axes_bool = np.zeros(self.lattice.ndim, dtype=bool)
        axes_bool[list(self.axes)] = True
        in_group = np.logical_and(self.lattice.pbc, axes_bool)
        shape = np.where(in_group, self.lattice.extent, 1)
        return shape

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

    def momentum_irrep(self, *k: Array) -> np.ndarray:
        r"""Returns the irrep characters (phase factors) corresponding to
        crystal momentum :math:`\vec k`."""
        # switch to reciprocal lattice coordinates
        k = self.lattice.to_reciprocal_lattice(_ensure_iterable(k)).squeeze()

        # prune axes with no nontrivial translations for performance
        shape = np.asarray(self.group_shape)
        nontrivial_axes = shape > 1
        k = k[nontrivial_axes]
        shape = shape[nontrivial_axes]

        # phase factors for translations along each axis
        axis_factors = [
            np.exp(-2j * np.pi * ki * np.arange(ni) / ni) for ki, ni in zip(k, shape)
        ]
        axis_factors = np.ix_(*axis_factors)

        return reduce(np.multiply, axis_factors).ravel()


_tg_efficiency_notice = """

        Computed more efficiently than for a generic
        :class:`~netket.utils.group.PermutationGroup` exploiting the
        Abelian group structure."""
TranslationGroup.inverse.__doc__ = (
    PermutationGroup.inverse.__doc__ + _tg_efficiency_notice
)
TranslationGroup.product_table.__doc__ = (
    PermutationGroup.product_table.__doc__ + _tg_efficiency_notice
)


@struct.dataclass
class SpaceGroup(PermutationGroup):
    """
    Class to handle the space group symmetries of a :class:`~netket.graph.Lattice`.

    Can be used as a :class:`~netket.utils.group.PermutationGroup`
    representing the action of a space group on a :class:`~netket.graph.Lattice`.
    The space group is generated as the semidirect product of the translation group
    of the lattice and a geometrical :class:`~netket.utils.group.PointGroup`
    given as a constructor argument.

    Also generates :class:`~netket.utils.group.PermutationGroup` representations of

    * the supplied point group,
    * its rotational subgroup (i.e. point group symmetries with determinant +1)
    * the translation group of the lattice

    as well as space group irreps for symmetrising wave functions.
    """

    lattice: Lattice
    """The lattice underlying the space group."""
    _point_group: PointGroup
    """The geometric point group underlying the space group."""
    point_group: PermutationGroup
    """The point group as a :class:`~netket.utils.group.PermutationGroup`
    acting on the sites of :attr:`lattice`.

    Group elements are listed in the order they appear in :attr:`_point_group`.
    Computed from :attr:`_point_group` upon construction, must not be changed after."""
    full_translation_group: TranslationGroup

    def __pre_init__(
        self, lattice: Lattice, point_group: PointGroup
    ) -> tuple[tuple, dict]:
        """
        Constructs the space group generated by the translation symmetries of
        the lattice and a given point group.

        Args:
            lattice: :class:`~netket.graph.Lattice`
                The lattice on which the space group is to act.
            point_group: :class:`~netket.utils.group.PointGroup`
                The geometrical point group underlying the space group.
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
        as a :class:`~netket.utils.group.PermutationGroup` acting on the sites of :attr:`lattice`.

        Group elements are listed in the order they appear in :attr:`_point_group`."""
        return _pg_to_permutation(self.lattice, self._point_group.rotation_group())

    def translation_group(
        self, axes: int | Sequence[int] | None = None
    ) -> TranslationGroup:
        """
        The group of valid translations of :attr:`lattice` as a :class:`TranslationGroup`
        acting on the sites of the same.
        """
        if axes is None:
            return self.full_translation_group
        else:
            return TranslationGroup(self.lattice, axes=axes)

    @deprecated_new_name("translation_group")
    def _translations_along_axis(self, axis: int) -> TranslationGroup:
        # DEPRECATED: use `self.translation_group(axes=axis)` instead
        return self.translation_group(axes=axis)

    @property
    @deprecated(
        reason="This `SpaceGroup` object can be used directly as a permutation group"
    )
    def space_group(self) -> "SpaceGroup":
        """
        Deprecated. Returns :code:`self`.
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

    @struct.property_cached
    def _point_group_conjugacy_table(self) -> np.ndarray:
        """Part of the conjugacy table :math:`h^{-1}gh` where h are
        point-group symmetries."""
        n_PG = len(self.point_group)
        col_index = np.arange(n_PG)[np.newaxis, :]
        # exploits that h^{-1}gh = (g^{-1} h)^{-1} h
        return self.product_table[self.product_table[:, :n_PG], col_index]

    def _little_group_index(self, k: Array) -> Array:
        r"""
        Returns the indices of the elements of the little group corresponding to
        wave vector :math:`\vec{k}`.
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
        r"""
        Returns the little co-group corresponding to wave vector :math:`\vec{k}`.
        This is the subgroup of the point group that leaves :math:`\vec{k}` invariant.

        Arguments:
            k: the wave vector in Cartesian axes

        Returns:
            the little co-group
        """
        k = _ensure_iterable(k)
        return PointGroup(
            [self._point_group[i] for i in self._little_group_index(k)],
            ndim=self._point_group.ndim,
            unit_cell=self.lattice.basis_vectors,
        )

    def little_group_multipliers(self, *k: Array) -> np.ndarray | None:
        r"""Computes the Schur multiplier associated with the little group
        given the translations associated with its elements.

        The mutlipliers are given by (Bradney & Cracknell, eqs. 3.7.11-14)

        .. math::

            \mu(S_i, S_j) &= \exp(-i g_i \cdot w_j)

            g_i  &= S_i^{-1} k - k

        and :math:`w_j` is the translation associated with
        point-group symmetry :math:`S_i`.

        Arguments:
            k: the wave vector in Cartesian axes

        Returns:
            A square array of the :math:`\mu(S_i, S_j)`.

            If all multipliers are +1, :code:`None` is returned instead:
            this signals to :meth:`~netket.utils.group.FiniteGroup.character_table()`
            etc. that linear (not projective) representations are required.
        """
        k = _ensure_iterable(k)
        ix = self._little_group_index(k)

        # g_i  = S_i^{-1} k - k
        matrices = self._point_group.matrices()[ix]
        matrices = matrices.transpose(0, 2, 1)  # need S_i^{-1}
        g = matrices @ k - k

        w = self._point_group.translations()[ix]

        multiplier = np.exp(-1j * (g @ w.T))

        if np.allclose(multiplier, 1.0, rtol=1e-8):
            return None
        else:
            return prune_zeros(multiplier)

    def little_group_irreps_readable(self, *k: Array, full: bool = False):
        """Returns a conventional rendering of little-group irrep characters.

        This differs from :code:`little_group(k).character_table_readable()`
        in that nontrivial Schur multipliers for nonsymmorphic space group
        are automatically taken into account.

        Arguments:
            k: the wave vector in Cartesian axes
            full: whether the character table for all group elements (True)
                or one representative per conjugacy class (False, default)

        Returns:

            A tuple containing a list of strings and an array

            - :code:`classes`: a text description of a representative of
              each conjugacy class (or each element) of the little group as a list
            - :code:`characters`: a matrix, each row of which lists the
              characters of one irrep
        """
        k = _ensure_iterable(k)
        group = self.little_group(k)
        multiplier = self.little_group_multipliers(k)

        if multiplier is not None:
            warn(
                "The space group is nonsymmorphic and the function will return\n"
                "a character table of projective irreps of the little group.\n"
                "If you want the linear irreps of the little group, use\n"
                "`self.little_group(k).character_table_readable()` instead."
            )

        return group.character_table_readable(multiplier, full)

    def _little_group_irreps(self, k: Array) -> Array:
        """
        Returns the character table of the little group embedded in the full point
        group. Symmetries outside the little group get 0.
        """
        idx = self._little_group_index(k)
        group = self.little_group(k)
        multiplier = self.little_group_multipliers(k)
        CT = group.character_table(multiplier)
        CT_full = np.zeros((CT.shape[0], len(self._point_group)), dtype=CT.dtype)
        CT_full[:, idx] = CT
        return CT_full

    def space_group_irreps(self, *k: Array) -> Array:
        r"""
        Returns the portion of the character table of the full space group
        corresponding to the star of the wave vector :math:`\vec{k}`.

        Arguments:
            k: the wave vector in Cartesian axes

        Returns:
            An array :code:`CT` listing the characters for all irreps of the
            space group defined on the star of :math:`\vec{k}`.

            :code:`CT[i]` returns the irrep corresponding to the little-group
            irrep listed in row #i by :meth:`little_group_irreps_readable`.

            :code:`CT[i,j]` gives the character of :code:`self[j]` in the same.
        """
        # One-arm irreps for the other wave vectors in the star can be
        # obtained from the arm `k` by conjugating the character with some
        # point-group operation.
        # The simplest is to do the conjugation with all of them, which
        # counts every arm |little group| times, which we divide out
        # at the end.
        k = _ensure_iterable(k)
        chi = self.one_arm_irreps(k)[:, self._point_group_conjugacy_table]
        return prune_zeros(chi.sum(axis=-1) / len(self._little_group_index(k)))

    def one_arm_irreps(self, *k: Array) -> Array:
        r"""
        Returns the portion of the character table of the full space group
        corresponding to the star of the wave vector :math:`\vec{k}`,
        projected onto :math:`\vec{k}` itself.

        Arguments:
            k: the wave vector in Cartesian axes

        Returns:
            An array `CT` listing the projected characters for all irreps of the
            space group defined on the star of :math:`\vec{k}`.

            :code:`CT[i]` returns the irrep corresponding to the little-group
            irrep listed in row #i by :meth:`little_group_irreps_readable`.

            :code:`CT[i,j]` gives the character of :code:`self[j]` in the same.
        """
        k = _ensure_iterable(k)
        # Little-group irrep factors
        # Phase factor for non-symmorphic symmetries is exp(-i w_g . p(k))
        point_group_factors = self._little_group_irreps(k) * np.exp(
            -1j * (self._point_group.translations() @ k)
        )
        # Translational factors
        trans_factors = self.full_translation_group.momentum_irrep(k)

        # Multiply the factors together
        # Translations are more major than point group operations
        result = np.einsum("ig, t -> itg", point_group_factors, trans_factors)
        result = result.reshape(point_group_factors.shape[0], -1)
        return prune_zeros(result)


_sg_efficiency_notice = """

        Computed more efficiently than for a generic
        :class:`~netket.utils.group.PermutationGroup` exploiting the
        semidirect product structure of space groups."""
SpaceGroup.product_table.__doc__ = (
    PermutationGroup.product_table.__doc__ + _sg_efficiency_notice
)
