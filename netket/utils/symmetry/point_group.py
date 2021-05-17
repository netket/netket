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

from plum import dispatch
import numpy as np
from dataclasses import dataclass

from .semigroup import Identity, Element
from .group import Group

from netket.utils import HashableArray, comparable
from netket.utils.types import Array, DType, Shape


class PGSymmetry(Element):
    def __init__(self, M: Array):
        """
        Creates a `PGSymmetry` object from an orthogonal matrix.

        Arguments:
            M: a 2D array specifying the transformation. It has to be orthogonal (i.e., it must be real and satisfy :math:`M^T M = \mathbf{1}`)

        Returns:
            A PGSymmetry object encoding the same transformation
        """
        # TODO add tests for ortohgonality?
        self.M = np.asarray(M)

    def __call__(self, x):
        return x @ M.T

    def __hash__(self):
        return hash(HashableArray(comparable(self.M)))
    
    def __eq__(self, other):
        if isinstance(other, Permutation):
            return HashableArray(comparable(self.M)) == HashableArray(comparable(other.M))
        else:
            return False

    def __repr__(self):
        # TODO use cleverer naming scheme
        return f"PGSymmetry({self.M})"

    def __array__(self, dtype: DType = None):
        return np.asarray(self.M, dtype)


@dispatch
def product(p: PGSymmetry, q: PGSymmetry):
    return PGSymmetry(p.M @ q.M)


@dataclass(frozen=True)
class PointGroup(Group):
    """
    Collection of point group symmetries acting on n-dimensional vectors.

    Group elements need not all be of type :ref:`netket.utils.symmetry.PGSymmetry`,
    only act on such vectors when called. Currently, however, only `Identity` and
    `PGSymmetry` have canonical forms implemented.

    The class can contain elements that are distinct as objects (e.g.,
    :code:`Identity()` and :code:`Rotation(0)`) but have identical action.
    Those can be removed by calling :code:`remove_duplicates`.
    """

    dim: int
    """Dimensionality of point group operations."""

    def __post_init__(self):
        super().__post_init__()
        
        # Check if all dimensionalities are correct 
        for x in self.elems:
            if isinstance(x, PGSymmetry) and (x.M.shape != (self.dim,self.dim)):
                raise ValueError("`PointGroup` contains operation of unexpected dimensionality")
            
        # Define custom hash
        myhash = hash((super().__hash__(), hash(self.dim)))
        object.__setattr__(self, "_PointGroup__hash", myhash)

    
    def __matmul__(self, other) -> "PointGroup":
        if not isinstance(other, PointGroup):
            raise ValueError(
                "Incompatible groups (`PointGroup` and something else)"
            )

        # Should check if dimensions match, but mismatched dimensions would lead to
        # multiplying different-sized matrices, resulting in an error
        return PointGroup(super().__matmul__(other).elems, self.dim)

    def _transformation_matrix(self, x: Element) -> Array:
        if isinstance(x, Identity):
            M = np.eye(self.dim, dtype=float)
        elif isinstance(x, PGSymmetry):
            M = x.M
        else:
            raise ValueError(
                "`PointGroup` only supports `Identity` and `PGSymmetry` elements"
            )
        return M
    
    def _canonical(self, x: Element) -> Array:
        return comparable(self._transformation_matrix(x))

    def to_array(self) -> Array:
        """
        Convert the abstract group operations to an array of transformation matrices
        such that `self.to_array()[i]` contains the transformation matrix of the
        `i`th group element.
        """
        return np.asarray([self._transformation_matrix(x) for x in self.elems])

    def __array__(self, dtype=None) -> Array:
        return np.asarray(self.to_array(), dtype=dtype)

    def remove_duplicates(self, *, return_inverse=False) -> "PointGroup":
        """
        Returns a new :code:`PointGroup` with duplicate elements (that is, elements which
        represent identical transformations) removed.

        Arguments:
            return_inverse: If True, also return indices to reconstruct the original
                group from the result.

        Returns:
            group: the point group with duplicate elements removed.
            return_inverse: Indices to reconstruct the original group from the result.
                Only returned if `return_inverse` is True.
        """
        if return_inverse:
            group, inverse = super().remove_duplicates(return_inverse=True)
        else:
            group = super().remove_duplicates(return_inverse=False)

        pgroup = PointGroup(group.elems, self.dim)

        if return_inverse:
            return pgroup, inverse
        else:
            return pgroup

    def _inverse(self) -> Array:
        lookup = self._canonical_lookup()
        
        inverse = np.zeros(len(self.elems), dtype=int)
        
        for index, element in enumerate(self.elems):
            # we exploit that the inverse of an orthogonal matrix is its transpose
            inverse_matrix = self._transformation_matrix(element).T
            inverse[index] = lookup[HashableArray(comparable(inverse_matrix))]
            
        return inverse

    def _product_table(self) -> Array:
        # again, we calculate the product table of transformation matrices directly
        trans_matrices = self.to_array()
        product_matrices = np.einsum('iab, jac -> ijbc', trans_matrices, trans_matrices) # this is a table of M_g^{-1} M_h = M_g.T M_h
        product_matrices = comparable(product_matrices)
        
        lookup = self._canonical_lookup()

        n_symm = len(self)
        product_table = np.zeros((n_symm, n_symm), dtype=int)

        for i in range(n_symm):
            for j in range(n_symm):
                product_table[i,j] = lookup[HashableArray(product_matrices[i,j])]

        return product_table

    @property
    def shape(self) -> Shape:
        """Tuple `(<# of group elements>, <dim>, <dim>)`, same as :code:`self.to_array().shape`."""
        return (len(self), self.dim, self.dim)

    def __hash__(self):
        # pylint: disable=no-member
        return self.__hash
