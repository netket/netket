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
from typing import overload, Literal, Any

import numpy as np

from netket.utils import HashableArray, struct
from netket.utils.float import comparable, prune_zeros
from netket.utils.types import Array
from netket.utils.dispatch import dispatch

from netket.utils.group._semigroup import Element, FiniteSemiGroup, Identity


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

    @overload
    def remove_duplicates(
        self, *, return_inverse: Literal[False] = False
    ) -> "FiniteGroup": ...

    @overload
    def remove_duplicates(
        self, *, return_inverse: Literal[True]
    ) -> tuple["FiniteGroup", Any]: ...

    def remove_duplicates(self, *, return_inverse=False):
        r"""
        Returns a new :class:`FiniteGroup` with duplicate elements (that is,
        elements with identical canonical forms) removed.

        Arguments:
            return_inverse: If True, also return indices to reconstruct the
                original group from the result.

        Returns:
            The group with duplicate elements removed. If
            :code:`return_inverse==True` it also returns the list of indices
            needed to reconstruct the original group from the result.
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

        Assuming the definitions

        .. code::

            g = self[idx_g]
            h = self[self.inverse[idx_g]]

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

        Assuming the definitions

        .. code::

            g = self[idx_g]
            h = self[idx_h]
            idx_u = self.product_table[idx_g, idx_h]

        :code:`self[idx_u]` corresponds to :math:`u = g^{-1} h` .
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
        The conjugacy table of the group.

        Assuming the definitions

        .. code::

            g = self[idx_g]
            h = self[idx_h]

        :code:`self[self.conjugacy_table[idx_g,idx_h]]`
        corresponds to :math:`h^{-1}gh`.
        """
        col_index = np.arange(len(self))[np.newaxis, :]
        # exploits that h^{-1}gh = (g^{-1} h)^{-1} h
        return self.product_table[self.product_table, col_index]

    @struct.property_cached
    def conjugacy_classes(self) -> tuple[Array, Array, Array]:
        r"""
        The conjugacy classes of the group.

        Returns:

            The three arrays

            - classes: a boolean array, each row indicating the elements that
              belong to one conjugacy class
            - representatives: the lowest-indexed member of each conjugacy class
            - inverse: the conjugacy class index of every group element

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

    def check_multiplier(self, multiplier: Array, rtol=1e-8, atol=0) -> bool:
        r"""
        Checks the associativity constraint of Schur multipliers.

        .. math::

            \alpha(x, y) \alpha(xy, z) = \alpha(x, yz) \alpha(y, z).


        Arguments:
            multiplier: the array of Schur multipliers :math:`\alpha(x,y)`
            rtol: relative tolerance
            atol: absolute tolerance

        Returns:
            whether :code:`multiplier` is a valid Schur multiplier
            up to the given tolerance

        Raises:
            :class:`ValueError`
                if the shape of `multiplier` does not match
                the size of the group
        """
        if multiplier.shape != (len(self), len(self)):
            raise ValueError(
                "Schur multipliers must form a square array of the same size\n"
                f"as the group ({len(self)}, {len(self)}), "
                f"got {multiplier.shape} instead!"
            )

        PT = self.product_table[self.inverse]
        # \alpha(x, y) \alpha(xy, z)
        left = np.expand_dims(multiplier, 2) * multiplier[PT, :]
        # \alpha(x, yz) \alpha(y, z)
        right = multiplier[:, PT] * np.expand_dims(multiplier, 0)
        return np.allclose(left, right, rtol=rtol, atol=atol)

    def _character_from_class_matrix(
        self, class_matrix: Array, which_class: Array | None = None
    ) -> np.ndarray:
        r"""Given a linear combination of class matrices,
        diagonalise it and normalise the eigenvectors as characters:

        .. math::

            \sum_g |\chi(g)|^2 = \sum_C |C| |\chi(c_0)|^2 = |G|.

        Arguments:
            class_matrix: square matrix diagonalised by the characters
                :math:`\chi(c_0)` (without any further scaling).
            which_class: (optional) boolean array, specifying which
                conjugacy classes are represented in `class_matrix`.

        Returns:
            irrep characters obtained from the eigenvectors of `class_matrix`.

            Excluded classes are filled in with zeros. The irreps are
            sorted by dimension.
        """
        _, table = np.linalg.eig(class_matrix)
        table = table.T  # want eigenvectors as rows

        # normalisation
        classes, _, _ = self.conjugacy_classes
        class_sizes = classes.sum(axis=1)
        if which_class is not None:
            class_sizes = class_sizes[which_class]
        norm = np.sum(np.abs(table) ** 2 * class_sizes, axis=1, keepdims=True)
        norm = (norm / len(self)) ** 0.5
        table /= norm
        # ensure correct sign (i.e. identity should have a real character)
        table /= _cplx_sign(table[:, 0])[:, np.newaxis]

        # Sort lexicographically, ascending by first column, descending by others
        sorting_table = np.column_stack((table.real, table.imag))
        sorting_table[:, 1:] *= -1
        sorting_table = comparable(sorting_table)
        _, indices = np.unique(sorting_table, axis=0, return_index=True)
        table = table[indices]

        # Get rid of annoying nearly-zero entries
        table = prune_zeros(table)

        if which_class is not None:
            # fill in excluded classes
            full_table = np.zeros((len(table), len(classes)), dtype=table.dtype)
            full_table[:, which_class] = table
            return full_table
        else:
            return table

    def projective_characters_by_class(
        self, multiplier: Array | None
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Calculates the character table of projective representations with a
        given Schur multiplier α using a modified Burnside algorithm.

        Arguments:
            multiplier: the unitary Schur multiplier.
                If unspecified, computes linear representation characters.

        Returns:
            - :code:`characters_by_class`
                a 2D array, each row containing the characters of a
                representative element of each conjugacy class in one
                projective irrep with the given multiplier.
            - :code:`class_factors`
                a 1D array listing the "class factors" of each element of
                the group. The character of each element is the product
                of the character of the class representative with this
                class factor.
                (Only returned if :code:`multiplier` is not :code:`None`.)

        Note: the algorithm and the definitions above are explained in more
        detail in https://arxiv.org/abs/2505.14790.
        """
        # deal with trivial multipliers
        if multiplier is None:
            return self.character_table_by_class
        elif np.allclose(multiplier, 1.0, rtol=0, atol=1e-8):
            # still trivial but should conform to projective irrep return format
            return self.character_table_by_class, np.ones(len(self))

        # check unitarity
        if not np.allclose(np.abs(multiplier), 1.0, rtol=0, atol=1e-8):
            raise ValueError("Schur multiplier must be unitary")

        # compute class factors
        class_factors = np.zeros(len(self), dtype=multiplier.dtype)
        classes, class_reps, _ = self.conjugacy_classes
        reg_classes = np.zeros(len(class_reps), dtype=bool)

        numerator = multiplier[self.inverse, np.arange(len(self))]
        for i, (g, cls) in enumerate(zip(class_reps, classes)):
            # β(g, h^-1 g h) = α(h^-1, h) / α(h^-1, g) α(h^-1 g, h)
            β = (
                numerator
                / multiplier[self.inverse, g]
                / multiplier[self.product_table[:, g], np.arange(len(self))]
            )

            class_size = cls.sum()
            centralizer_size = len(self) // class_size
            # collate sets of h with equal h^-1 g h
            h_sets = np.argsort(self.conjugacy_table[g]).reshape(
                class_size, centralizer_size
            )
            # average β across each set
            β = np.average(β[h_sets], axis=1)

            if np.allclose(np.abs(β), 1.0, rtol=0, atol=1e-8):
                # if the class is α-regular, β for each averaged entry was equal
                # and the array now contains unit complex numbers
                class_factors[cls] = β
                reg_classes[i] = True
            elif not np.allclose(np.abs(β), 0.0, rtol=0, atol=1e-8):
                # otherwise, the different β should average to zero
                raise RuntimeError(
                    "Class factors close to neither unity of zero\n" + repr(β)
                )

        # The algorithm hinges on the equation
        #   \sum_C M_{ABC} \chi(c_0) = |A||B| \chi(a_0) \chi(b_0) / d_\chi,
        # where
        #   M_{ABC} = \sum_{a \in A, b \in B, ab \in C} α(a,b) β(ab)/β(a)β(b)
        #
        # From the oblique times table, we can easily extract the array
        #   (M'_B)_{AC} = \sum_{a \in A, ab_0 \in C} α(a, b_0) β(ab_0)/β(a)
        #               = M_{ABC} / |B|
        # by building an array of α(a, b_0) β(ab_0)/β(a), filtering for
        # `product_table == b_0`, and aggregating over conjugacy classes.
        #
        # Defining
        #   (M_B)_{AC} := (M'_B)_{AC} / |A|,
        # we finally have the eigenvalue equation
        #   M_B \vec\chi = \chi(b_0)/d_\chi \vec\chi,
        # where \vec\chi is the vector of characters \chi(c_0).

        # Construct a random linear combination of the M_B
        # array of α(a,b) β(ab)/β(a)
        class_matrix = (
            multiplier[np.arange(len(self))[:, None], self.product_table]
            * class_factors
            # set non-regular elements to zero instead of inf/nan
            / np.where(class_factors == 0.0, np.inf, class_factors)[:, None]
        )

        # multiply with random weights for α-regular class representatives
        # TODO should we have a term for every α-regular element?
        weight_ = random(reg_classes.sum(), seed=0, cplx=np.iscomplexobj(multiplier))
        weight = np.zeros(len(self), dtype=weight_.dtype)
        weight[class_reps[reg_classes]] = weight_
        class_matrix *= weight[self.product_table]

        # aggregate by α-regular conjugacy class, divide by |A|
        classes = classes[reg_classes]
        class_matrix = classes @ class_matrix @ classes.T
        class_matrix /= classes.sum(axis=1)[:, None]

        return (
            self._character_from_class_matrix(class_matrix, reg_classes),
            class_factors,
        )

    @struct.property_cached
    def character_table_by_class(self) -> np.ndarray:
        r"""
        Calculates the character table using Burnside's algorithm.

        Each row of the output lists the characters of one irrep in the order the
        conjugacy classes are listed in :attr:`conjugacy_classes`.

        Assumes that :code:`Identity() == self[0]`, if not, the sign of some
        characters may be flipped. The irreps are sorted by dimension.
        """
        classes, _, _ = self.conjugacy_classes
        class_sizes = classes.sum(axis=1)
        # Burnside's algorithm hinges on the equation
        #   \sum_C M_{ABC} \chi_C = |A||B| \chi_A \chi_B / d_\chi,
        # where
        #   M_{ABC} = #{a \in A, b \in B: ab \in C}.
        #
        # From the oblique times table, we can easily extract the array
        #   (M'_B)_{AC} = #{a \in A: ab_0 \in C}
        # by aggregating `product_table == b_0` over conjugacy classes.
        #
        # Defining
        #   (M_B)_{AC} := (M'_B)_{AC} / |A|,
        # we finally have the eigenvalue equation
        #   M_B \vec\chi = \chi_B / d_\chi \vec\chi,
        # where \vec\chi is the vector of characters \chi_C.

        # Construct a random linear combination of the M_B
        # i.e. weight each `product_table==b` with some random weight w(b)
        class_matrix = (
            classes @ random(len(self), seed=0)[self.product_table] @ classes.T
        )
        class_matrix /= class_sizes[:, None]

        return self._character_from_class_matrix(class_matrix)

    def character_table(self, multiplier: Array | None = None) -> np.ndarray:
        r"""
        Calculates the character table using Burnside's algorithm.

        Arguments:
            multiplier: (optional) Schur multiplier

        Returns:
            a matrix of all linear irrep characters (if :code:`multiplier is None`)
            or projective irrep characters with the given :code:`multiplier`,
            sorted by dimension.

            Each row of lists the characters of all group elements
            for one irrep, i.e. :code:`self.character_table()[i,g]`
            gives :math:`\chi_i(g)`.

        It is assumed that :code:`Identity() == self[0]`. If not, the sign
        of some characters may be flipped and the sorting by dimension
        will be wrong.
        """
        _, _, class_idx = self.conjugacy_classes
        if multiplier is None or np.allclose(multiplier, 1.0, rtol=0, atol=1e-8):
            # linear representations
            CT = self.character_table_by_class
            return CT[:, class_idx]
        else:
            # projective representations
            CT, class_factor = self.projective_characters_by_class(multiplier)
            return CT[:, class_idx] * class_factor

    def character_table_readable(
        self, multiplier: Array | None = None, full: bool = False
    ) -> tuple[list[str], Array]:
        r"""
        Returns a conventional rendering of the character table.

        Arguments:
            multiplier: (optional) Schur multiplier
            full: whether the character table for all group elements (True)
                or one representative per conjugacy class (False, default)

        Returns:

            A tuple containing a list of strings and an array

            - :code:`classes`: a text description of a representative of
              each conjugacy class (or each group element) as a list
            - :code:`characters`: a matrix, each row of which lists the
              characters of one irrep
        """
        if full:
            names = [str(g) for g in self]
            return names, self.character_table(multiplier)
        else:
            classes, idx_repr, _ = self.conjugacy_classes
            class_sizes = classes.sum(axis=1)
            representatives = [
                f"{class_sizes[cls]}x{self[rep]}" for cls, rep in enumerate(idx_repr)
            ]
            if multiplier is None or np.allclose(multiplier, 1.0, rtol=0, atol=1e-8):
                # linear representations
                CT = self.character_table_by_class
            else:
                # projective representations
                CT, _ = self.projective_characters_by_class(multiplier)
            return representatives, CT

    @struct.property_cached
    def _irrep_matrices(self) -> list[Array]:
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

    def irrep_matrices(self) -> list[Array]:
        """
        Returns matrices that realise all irreps of the group.

        Returns:
            A list of 3D arrays such that :code:`self.irrep_matrices()[i][g]`
            contains the representation of :code:`self[g]` consistent with
            the characters in :code:`self.character_table()[i]`.
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
