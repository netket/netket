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

import itertools
from typing import List, Tuple

import numpy as np

from netket.utils import HashableArray, struct
from netket.utils.float import comparable, prune_zeros
from netket.utils.types import Array
from netket.utils.dispatch import dispatch

from ._semigroup import Element, FiniteSemiGroup, Identity


@struct.dataclass
class FiniteGroup(FiniteSemiGroup):
    """
    Collection of Elements expected to satisfy group axioms.
    Unlike FiniteSemiGroup, product tables, conjugacy classes, etc. can be calculated.

    Group elements can be implemented in any way, as long as a subclass of Group
    is able to implement their action. Subclasses must implement a :code:`_canonical()`
    method that returns an array of integers for each acceptable Element such that
    two Elements are considered equal iff the corresponding matrices are equal.

    """

    def __hash__(self):
        return super().__hash__()

    def _canonical(self, x: Element) -> Array:
        r"""
        Canonical form of :code:`Element`s, used for equality testing (i.e., two
        :code:`Element`s `x,y` are deemed equal iff
        :code:`_canonical(x) == _canonical(y)`.
        Must be overridden in subclasses

        Arguments:
            x: an `Element`

        Returns:
            the canonical form as a numpy.ndarray of integers
        """
        raise NotImplementedError

    def _canonical_array(self) -> Array:
        r"""
        Lists the canonical forms returned by `_canonical` as rows of a 2D array.
        """
        return np.array([self._canonical(x).flatten() for x in self.elems])

    def _canonical_lookup(self) -> dict:
        r"""
        Creates a lookup table from canonical forms to index in `self.elems`
        """
        return {
            HashableArray(self._canonical(element)): index
            for index, element in enumerate(self.elems)
        }

    def remove_duplicates(self, *, return_inverse=False) -> "FiniteGroup":
        r"""
        Returns a new :code:`FiniteGroup` with duplicate elements (that is,
        elements with identical canonical forms) removed.

        Arguments:
            return_inverse: If True, also return indices to reconstruct the original
                group from the result.

        Returns:
            group: the group with duplicate elements removed.
            inverse: Indices to reconstruct the original group from the result.
                Only returned if `return_inverse` is True.
        """
        result = np.unique(
            self._canonical_array(),
            axis=0,
            return_index=True,
            return_inverse=return_inverse,
        )
        group = FiniteGroup([self.elems[i] for i in sorted(result[1])])
        if return_inverse:
            return group, result[2]
        else:
            return group

    @struct.property_cached
    def inverse(self) -> Array:
        r"""
        Indices of the inverse of each element.
        If :code:`g = self[idx_g]` and :code:`h = self[self.inverse[idx_g]]`, then
        :code:`gh = product(g, h)` is equivalent to :code:`Identity()`
        """
        canonical_identity = self._canonical(Identity())
        inverse = np.zeros(len(self.elems), dtype=int)

        for i, e1 in enumerate(self.elems):
            for j, e2 in enumerate(self.elems):
                prod = e1 @ e2
                if np.all(self._canonical(prod) == canonical_identity):
                    inverse[i] = j

        return inverse

    @struct.property_cached
    def product_table(self) -> Array:
        r"""
        A table of indices corresponding to :math:`g^{-1} h` over the group.
        That is, if :code:`g = self[idx_g]', :code:`h = self[idx_h]`, and
        :code:`idx_u = self.product_table[idx_g, idx_h]`, then :code:`self[idx_u]`
        corresponds to :math:`u = g^{-1} h`.
        """
        n_symm = len(self)
        product_table = np.zeros([n_symm, n_symm], dtype=int)

        lookup = self._canonical_lookup()

        for i, e1 in enumerate(self.elems[self.inverse]):
            for j, e2 in enumerate(self.elems):
                prod = e1 @ e2
                product_table[i, j] = lookup[HashableArray(self._canonical(prod))]

        return product_table

    @struct.property_cached
    def conjugacy_table(self) -> Array:
        r"""
        A table of conjugates: if `g = self[idx_g]` and `h = self[idx_h]`,
        then `self[self.conjugacy_table[idx_g,idx_h]]` is :math:`h^{-1}gh`.
        """
        col_index = np.arange(len(self))[np.newaxis, :]
        # exploits that h^{-1}gh = (g^{-1} h)^{-1} h
        return self.product_table[self.product_table, col_index]

    @struct.property_cached
    def conjugacy_classes(self) -> Tuple[Array, Array, Array]:
        r"""
        The conjugacy classes of the group.

        Returns:
            classes: a boolean array, each row indicating the elements that belong
                to one conjugacy class
            representatives: the lowest-indexed member of each conjugacy class
            inverse: the conjugacy class index of every group element
        """
        row_index = np.arange(len(self))[:, np.newaxis]

        # if is_conjugate[i,j] is True, self[i] and self[j] are in the same class
        is_conjugate = np.full((len(self), len(self)), False)
        is_conjugate[row_index, self.conjugacy_table] = True

        classes, representatives, inverse = np.unique(
            is_conjugate, axis=0, return_index=True, return_inverse=True
        )

        # Usually self[0] == Identity(), which is its own class
        # This class is listed last by the lexicographic order used by np.unique
        # so we reverse it to get a more conventional layout
        classes = classes[::-1]
        representatives = representatives[::-1]
        inverse = (representatives.size - 1) - inverse

        return classes, representatives, inverse

    @struct.property_cached
    def character_table_by_class(self) -> Array:
        r"""
        Calculates the character table using Burnside's algorithm.

        Each row of the output lists the characters of one irrep in the order the
        conjugacy classes are listed in `self.conjugacy_classes`.

        Assumes that `Identity() == self[0]`, if not, the sign of some characters
        may be flipped. The irreps are sorted by dimension.
        """
        classes, _, _ = self.conjugacy_classes
        class_sizes = classes.sum(axis=1)
        # Construct a random linear combination of the class matrices c_S
        #    (c_S)_{RT} = #{r,s: r \in R, s \in S: rs = t}
        # for conjugacy classes R,S,T, and a fixed t \in T.
        #
        # From our oblique times table it is easier to calculate
        #    (d_S)_{RT} = #{r,t: r \in R, t \in T: rs = t}
        # for a fixed s \in S. This is just `product_table == s`, aggregrated
        # over conjugacy classes. c_S and d_S are related by
        #    c_{RST} = |S| d_{RST} / |T|;
        # since we only want a random linear combination, we forget about the
        # constant |S| and only divide each column through with the appropriate |T|
        class_matrix = (
            classes @ random(len(self), seed=0)[self.product_table] @ classes.T
        )
        class_matrix /= class_sizes

        # The vectors |R|\chi(r) are (column) eigenvectors of all class matrices
        # the random linear combination ensures (with probability 1) that
        # none of them are degenerate
        _, table = np.linalg.eig(class_matrix)
        table = table.T / class_sizes

        # Normalise the eigenvectors by orthogonality: \sum_g |\chi(g)|^2 = |G|
        norm = np.sum(np.abs(table) ** 2 * class_sizes, axis=1, keepdims=True) ** 0.5
        table /= norm
        table /= _cplx_sign(table[:, 0])[:, np.newaxis]  # ensure correct sign
        table *= len(self) ** 0.5

        # Sort lexicographically, ascending by first column, descending by others
        sorting_table = np.column_stack((table.real, table.imag))
        sorting_table[:, 1:] *= -1
        sorting_table = comparable(sorting_table)
        _, indices = np.unique(sorting_table, axis=0, return_index=True)
        table = table[indices]

        # Get rid of annoying nearly-zero entries
        table = prune_zeros(table)

        return table

    def character_table(self) -> Array:
        r"""
        Calculates the character table using Burnside's algorithm.

        Each row of the output lists the characters of all group elements for one irrep,
        i.e. self.character_table()[i,g] gives :math:`\chi_i(g)`.

        Assumes that `Identity() == self[0]`, if not, the sign of some characters
        may be flipped. The irreps are sorted by dimension.
        """
        _, _, inverse = self.conjugacy_classes
        CT = self.character_table_by_class
        return CT[:, inverse]

    def character_table_readable(self) -> Tuple[List[str], Array]:
        r"""
        Returns a conventional rendering of the character table.

        Returns:
            classes: a text description of a representative of each conjugacy class
                as a list
            characters: a matrix, each row of which lists the characters of one irrep
        """
        # TODO put more effort into nice rendering?
        classes, idx_repr, _ = self.conjugacy_classes
        class_sizes = classes.sum(axis=1)
        representatives = [
            f"{class_sizes[cls]}x{self[rep]}" for cls, rep in enumerate(idx_repr)
        ]
        return representatives, self.character_table_by_class

    @struct.property_cached
    def _irrep_matrices(self) -> List[Array]:
        # We use Dixon's algorithm (Math. Comp. 24 (1970), 707) to decompose
        # the regular representation of the group into its irreps.
        # We start with a Hermitian matrix E that commutes with every matrix in
        # this rep: the space spanned by its degenerate eigenvectors then all
        # transform according to some rep of the group.
        # The matrix is randomised to ensure there are no accidental degeneracies,
        # i.e. all these spaces are irreps.
        # For real irreps, real matrices are returned: if needed, the same
        # routine is run with a real symmetric and a complex Hermitian matrix.

        true_product_table = self.product_table[self.inverse]
        inverted_product_table = true_product_table[:, self.inverse]

        def invariant_subspaces(e, seed):
            # Construct a Hermitian matrix that commutes with all matrices
            # in the regular rep.
            # These matrices obey E_{g,h} = e_{gh^{-1}} for some vector e
            e = e[inverted_product_table]
            e = e + e.T.conj()

            # Since E commutes with all the ρ, its eigenspaces reduce the rep
            # For a random input vector, there are no accidental degeneracies
            # except for complex irreps and real symmetric E.
            e, v = np.linalg.eigh(e)

            # indices that split v into eigenspaces
            _, starting_idx = np.unique(comparable(e), return_index=True)

            # Calculate sᴴPv for one eigenvector v per eigenspace, a fixed
            # random vector s and all regular projectors P
            # These are calculated as linear combinations of sᴴρv for the
            # regular rep matrices ρ, which is given by the latter two terms
            vs = v[:, starting_idx]
            s = random(len(self), seed=seed)[inverted_product_table]
            # row #i of this `s` is sᴴρ(self[i]), where sᴴ is the random vector
            proj = self.character_table().conj() @ s @ vs
            starting_idx = list(starting_idx) + [len(self)]
            return v, starting_idx, proj

        # Calculate the eigenspaces for a real and/or complex random matrix
        # To check which are needed, calculate Frobenius-Schur indicators
        # This is +1, 0, -1 for real, complex, and quaternionic irreps
        squares = np.diag(true_product_table)
        frob = np.array(
            np.rint(
                np.sum(self.character_table()[:, squares], axis=1).real / len(self)
            ),
            dtype=int,
        )
        eigen = {}
        if np.any(frob == 1):
            # real irreps: start from a real symmetric invariant matrix
            e = random(len(self), seed=0)
            eigen["real"] = invariant_subspaces(e, seed=1)
        if np.any(frob != 1):
            # complex or quaternionic irreps: complex hermitian invariant matrix
            e = random(len(self), seed=2, cplx=True)
            eigen["cplx"] = invariant_subspaces(e, seed=3)

        irreps = []

        for i, chi in enumerate(self.character_table()):
            v, idx, proj = eigen["real"] if frob[i] == 1 else eigen["cplx"]
            # Check which eigenspaces belong to this irrep
            proj = np.logical_not(np.isclose(proj[i], 0.0))
            # Pick the first eigenspace in this irrep
            first = np.arange(len(idx) - 1, dtype=int)[proj][0]
            v = v[:, idx[first] : idx[first + 1]]
            # v is now the basis of a single irrep: project the regular rep onto it
            irreps.append(np.einsum("gi,ghj ->hij", v.conj(), v[true_product_table, :]))

        return irreps

    def irrep_matrices(self) -> List[Array]:
        """
        Returns matrices that realise all irreps of the group.

        Returns:
            A list of 3D arrays such that `self.irrep_matrices()[i][g]` contains
            the representation of `self[g]` consistent with the characters in
            `self.character_table()[i]`.
        """
        return self._irrep_matrices


def _cplx_sign(x):
    return x / np.abs(x)


def random(n, seed, cplx=False):
    if cplx:
        v = np.random.default_rng(seed).normal(size=(2, n))
        v = v[0] + 1j * v[1]
        return v
    else:
        return np.random.default_rng(seed).normal(size=n)


@dispatch
def product(A: FiniteGroup, B: FiniteGroup):
    return FiniteGroup(
        elems=[a @ b for a, b in itertools.product(A.elems, B.elems)],
    )
