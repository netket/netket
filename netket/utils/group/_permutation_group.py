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

import numpy as np

from netket.utils import HashableArray, struct
from netket.utils.types import Array, DType, Shape
from netket.utils.dispatch import dispatch

from netket.utils import warn_deprecation, deprecated, deprecated_new_name

from ._group import FiniteGroup
from ._semigroup import Element


class Permutation(Element):
    def __init__(
        self,
        permutation: Array | None = None,
        *,  # change one line somewhere
        name: str | None = None,
        permutation_array: Array | None = None,
        inverse_permutation_array: Array | None = None,
    ):
        r"""
        Creates a `Permutation` from either the array of images
        `permutation_array` or preimages `inverse_permutation_array`.

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
            name: Optional, custom name for the permutation.
            permutation_array: 1D array listing
                :math:`g(x)` for all :math:`0\le x \le N-1`.
            inverse_permutation_array: 1D array listing
                :math:`g^{-1}(x)` for all :math:`0\le x \le N-1`.

        Returns:
            A `Permutation` object that encodes the specified permutation.
        """

        arg_list = [permutation, permutation_array, inverse_permutation_array]
        if sum([arg is not None for arg in arg_list]) != 1:
            raise ValueError(
                "Exactly one argument among `permutation`, `permutation_array` "
                "and `inverse_permutation_array` must be specified."
            )

        if permutation is not None:
            warn_deprecation(
                "The argument `permutation` is deprecated.\n\n"
                "In order to clarify notations, you should either pass "
                "the array of images `permutation_array` or "
                "preimages `inverse_permutation_array`."
            )
            inverse_permutation_array = permutation

        if permutation_array is not None:
            inverse_permutation_array = np.argsort(permutation_array)

        self._inverse_permutation_array = HashableArray(
            np.asarray(inverse_permutation_array)
        )

        self.__name = name

    def __hash__(self):
        return hash(self._inverse_permutation_array)

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return np.array_equal(self.permutation_array, other.permutation_array)
        else:
            return False

    @property
    def permutation_array(self):
        return np.asarray(np.argsort(self._inverse_permutation_array))

    @property
    def inverse_permutation_array(self):
        return np.asarray(self._inverse_permutation_array)

    @property
    @deprecated(
        "Deprecated in favor of `permutation.inverse_permutation_array` or "
        "`permutation.permutation_array`"
    )
    def permutation(self):
        return np.asarray(self._inverse_permutation_array)

    @property
    def _name(self):
        return self.__name

    def __repr__(self):
        if self._name is not None:
            return self._name
        else:
            return f"Permutation({self.permutation_array.tolist()})"

    @deprecated_new_name("permutation.inverse_permutation_array")
    def __array__(self, dtype: DType = None):
        return np.asarray(self._inverse_permutation_array, dtype)

    def apply_to_id(self, x: Array):
        """Returns the image of indices `x` under the permutation"""
        return self.permutation_array[x]


@dispatch
def product(p: Permutation, x: Array):
    # if p.permutation is a HashableArray and x is a jax Array
    # direct indexing fails, so we call np.asarray on it to extract the
    # wrapped array
    # TODO make indexing work with HashableArray directly
    return x[..., p.inverse_permutation_array]


@dispatch
def product(p: Permutation, q: Permutation):  # noqa: F811
    name = None if p._name is None and q._name is None else f"{p} @ {q}"
    inverse_permutation_array = q.inverse_permutation_array[p.inverse_permutation_array]
    return Permutation(inverse_permutation_array=inverse_permutation_array, name=name)


@struct.dataclass
class PermutationGroup(FiniteGroup):
    r"""
    Collection of permutation operations acting on sequences of length :code:`degree`.

    Group elements need not all be of type :class:`netket.utils.group.Permutation`,
    only act as such on a sequence when called.

    The class can contain elements that are distinct as objects (e.g.,
    :code:`Identity()` and :code:`Translation((0,))`) but have identical action.
    Those can be removed by calling
    :meth:`~netket.utils.group.PermutationGroup.remove_duplicates`.
    """

    degree: int
    """Number of elements the permutations act on."""

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other: "PermutationGroup") -> bool:
        if isinstance(other, PermutationGroup):
            return np.array_equal(self.to_array(), other.to_array())
        return False

    def _canonical(self, x: Element) -> Array:
        return x(np.arange(self.degree, dtype=int))

    def to_array(self) -> Array:
        r"""
        Convert the abstract group operations to an array of permutation indices.


        It returns a matrix where the `i`-th row contains the indices corresponding
        to the `i`-th group element. That is, :code:`self.to_array()[i, j]`
        is :math:`g_i^{-1}(j)`. Moreover,

        .. code::

            G = # this permutation group...
            V = np.arange(G.degree)
            assert np.all(G(V) == V[..., G.to_array()])

        Returns:
            A matrix that can be used to index arrays in the computational basis
            in order to obtain their permutations.
        """
        return self._canonical_array()

    def __array__(self, dtype=None) -> Array:
        return np.asarray(self.to_array(), dtype=dtype)

    def remove_duplicates(self, *, return_inverse=False) -> "PermutationGroup":
        r"""
        Returns a new :code:`PermutationGroup` with duplicate elements (that is,
        elements which represent identical permutations) removed.

        Args:
            return_inverse: If `True`, also return indices to reconstruct the original
                group from the result.

        Returns:
            The permutation group with duplicate elements removed. If
            :code:`return_inverse==True`, it also returns the indices needed to
            reconstruct the original group from the result.
        """
        if return_inverse:
            group, inverse = super().remove_duplicates(return_inverse=True)
        else:
            group = super().remove_duplicates(return_inverse=False)

        pgroup = PermutationGroup(group.elems, self.degree)

        if return_inverse:
            return pgroup, inverse
        else:
            return pgroup

    @struct.property_cached
    def inverse(self) -> Array:
        try:
            lookup = self._canonical_lookup()
            inverses = []
            # `np.argsort` on a 1D permutation list generates the inverse permutation
            # it acts along last axis by default, so can perform it on to_array()
            # `np.argsort` changes int32 to int64 on Windows,
            # and we need to change it back
            perms = self.to_array()
            invperms = np.argsort(perms).astype(perms.dtype)

            for invperm in invperms:
                inverses.append(lookup[HashableArray(invperm)])

            return np.asarray(inverses, dtype=int)
        except KeyError as err:
            raise RuntimeError(
                "PermutationGroup does not contain the inverse of all elements"
            ) from err

    @struct.property_cached
    def product_table(self) -> Array:
        perms = self.to_array()
        inverse = perms[self.inverse].squeeze()
        n_symm = len(perms)
        lookup = np.unique(np.column_stack((perms, np.arange(len(self)))), axis=0)

        product_table = np.zeros([n_symm, n_symm], dtype=int)
        for i, g_inv in enumerate(inverse):
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
            product_table[i, row_perms[:, -1]] = lookup[:, -1]

        return product_table

    @property
    def shape(self) -> Shape:
        r"""
        Tuple :code:`(<# of group elements>, <degree>)`.

        Equivalent to :code:`self.to_array().shape`.
        """
        return (len(self), self.degree)

    def apply_to_id(self, x: Array):
        """Returns the image of indices `x` under all permutations"""
        return self.to_array()[self.inverse][:, x]


@dispatch
def product(A: PermutationGroup, B: PermutationGroup):  # noqa: F811
    if A.degree != B.degree:
        raise ValueError(
            "Incompatible groups (`PermutationGroup`s of different degree)"
        )
    return PermutationGroup(
        elems=[a @ b for a, b in itertools.product(A.elems, B.elems)], degree=A.degree
    )


@dispatch
def product(G: PermutationGroup, x: Array):  # noqa: F811
    return np.moveaxis(x[..., G.to_array()], -2, 0)
