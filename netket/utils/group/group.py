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
from dataclasses import dataclass
from plum import dispatch
import itertools

from .semigroup import FiniteSemiGroup, Element, Identity
from netket.utils import struct, HashableArray
from netket.utils.float import comparable, prune_zeros
from netket.utils.types import Array, DType, Shape, PyTree
from typing import Tuple, List


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
        """
        Canonical form of :code:`Element`s, used for equality testing (i.e., two :code:`Element`s
        `x,y` are deemed equal iff :code:`_canonical(x) == _canonical(y)`.
        Must be overridden in subclasses

        Arguments:
            x: an `Element`

        Returns:
            the canonical form as a numpy.ndarray of integers
        """
        raise NotImplementedError

    def _canonical_array(self) -> Array:
        """
        Lists the canonical forms returned by `_canonical` as rows of a 2D array.
        """
        return np.array([self._canonical(x).flatten() for x in self.elems])

    def _canonical_lookup(self) -> dict:
        """
        Creates a lookup table from canonical forms to index in `self.elems`
        """
        return {
            HashableArray(self._canonical(element)): index
            for index, element in enumerate(self.elems)
        }

    def remove_duplicates(self, *, return_inverse=False) -> "Group":
        """
        Returns a new :code:`Group` with duplicate elements (that is, elements with
        identical canonical forms) removed.

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
        group = Group([self.elems[i] for i in sorted(result[1])])
        if return_inverse:
            return group, result[2]
        else:
            return group

    @struct.property_cached
    def inverse(self) -> Array:
        """
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
        """
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
        """
        A table of conjugates: if `g = self[idx_g]` and `h = self[idx_h]`,
        then `self[self.conjugacy_table[idx_g,idx_h]]` is :math:`h^{-1}gh`.
        """
        col_index = np.arange(len(self))[np.newaxis, :]
        # exploits that h^{-1}gh = (g^{-1} h)^{-1} h
        return self.product_table[self.product_table, col_index]

    @struct.property_cached
    def conjugacy_classes(self) -> Tuple[Array, Array, Array]:
        """
        The conjugacy classes of the group.

        Returns:
            classes: a boolean array, each row indicating the elements that belong to one conjugacy class
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
        """
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
            classes @ np.random.uniform(size=len(self))[self.product_table] @ classes.T
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
        """
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
        """
        Returns a conventional rendering of the character table.

        Returns:
            classes: a text description of a representative of each conjugacy class as a list
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
    def _irrep_matrices(self) -> PyTree:
        random = np.random.standard_normal
        # Generate the eigensystem of a random linear combination of matrices
        # in the group's regular representation (namely, product_table == g)
        # The regular representation contains d copies of each d-dimensional irrep
        # all of which return the same eigenvalues with eigenvectors that
        # correspond to one another in the bases of the several irrep copies
        e, vs = np.linalg.eig(random(len(self))[self.product_table])
        e = np.stack((e.real, e.imag))
        # we only need one eigenvector per eigenvalue
        _, idx = np.unique(comparable(e), return_index=True, axis=1)
        vs = vs[:, idx]
        # We will nedd the true product table as well
        true_product_table = self.product_table[self.inverse]

        irreps = []

        for chi in self.character_table():
            # Check which eigenvectors belong to this irrep
            # the last argument of einsum is the regular projector on this irrep
            proj = np.einsum(
                "gi,hi,gh->i", vs.conj(), vs, chi.conj()[self.product_table]
            )
            proj = np.logical_not(np.isclose(proj, 0.0))
            # Pick the first eigenvector in this irrep
            idx = np.arange(vs.shape[1], dtype=int)[proj][0]
            v = vs[:, idx]
            # Generate an orthonormal basis for the irrep spanned by A_reg(g).v
            # for all matrices in the regular representation
            # A basis follows (with probability 1) as d random linear
            # combinations of these vectors; make them orthonormal using QR
            # NB v[product_table] generates a matrix whose columns are A_reg(h).v
            dim = int(np.rint(chi[0].real))
            w, _ = np.linalg.qr(v[true_product_table] @ random((len(self), dim)))
            # Project the regular representation on this basis
            irreps.append(
                prune_zeros(
                    np.einsum("gi,ghj ->hij", w.conj(), w[true_product_table, :])
                )
            )

        return irreps

    def irrep_matrices(self) -> PyTree:
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


@dispatch
def product(A: FiniteGroup, B: FiniteGroup):
    return FiniteGroup(
        elems=[a @ b for a, b in itertools.product(A.elems, B.elems)],
    )
