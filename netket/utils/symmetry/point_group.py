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
from math import pi
from functools import partial

from .semigroup import Identity, Element
from .group import Group

from netket.utils import HashableArray, comparable, struct
from netket.utils.types import Array, DType, Shape

############ POINT GROUP SYMMETRY CLASS ########################################


@struct.dataclass
class PGSymmetry(Element):
    """
    An abstract group element object to geometrically describe point group symmetries
    that leave the origin in place geometrically.
    """

    M: Array
    """
    A 2D array specifying the transformation: a vector :math:`vec x` is mapped
    to :math:`M\vec x`. It has to be orthogonal (i.e. it must be real and 
    satisfy :math:`M^T M = 1`)
    """

    def __call__(self, x):
        return x @ M.T

    def __hash__(self):
        return hash(HashableArray(comparable(self.M)))

    def __eq__(self, other):
        if isinstance(other, Permutation):
            return HashableArray(comparable(self.M)) == HashableArray(
                comparable(other.M)
            )
        else:
            return False

    @struct.property_cached
    def _name(self) -> str:
        if self.M.shape == (2, 2):
            return _2D_name(self.M)
        elif self.M.shape == (3, 3):
            return _3D_name(self.M)
        else:
            return f"PGSymmetry({self.M})"

    def __repr__(self):
        return self._name

    def __array__(self, dtype: DType = None):
        return np.asarray(self.M, dtype)


@dispatch
def product(p: PGSymmetry, q: PGSymmetry):
    return PGSymmetry(p.M @ q.M)


############ NAMING 2D AND 3D POINT GROUP SYMMETRIES ###########################

_naming_tol = 1e-6
_naming_allclose = partial(np.allclose, atol=_naming_tol, rtol=0.0)
_naming_isclose = partial(np.isclose, atol=_naming_tol, rtol=0.0)


def _2D_name(M: Array) -> str:
    if M.shape != (2, 2):
        raise ValueError("This function names 2D symmetries")
    if _naming_isclose(np.linalg.det(M), 1.0):  # rotations
        if _naming_allclose(M, np.eye(2)):  # identity
            return "Id()"
        else:
            angle = np.arctan2(M[1, 0], M[0, 0])
            # in practice, all rotations are by integer degrees
            angle = int(np.rint(np.degrees(angle)))
            return f"Rot({angle}°)"

    elif _naming_isclose(np.linalg.det(M), -1.0):  # reflections
        axis = np.arctan2(M[1, 0], M[0, 0]) / 2
        axis = int(np.rint(np.degrees(axis)))
        return f"Refl({axis}°)"

    else:
        raise ValueError("M must be an orthogonal matrix")


def _3D_name(M: Array) -> str:
    if M.shape != (3, 3):
        raise ValueError("This function names 2D symmetries")
    if _naming_isclose(np.linalg.det(M), 1.0):  # rotations
        if _naming_allclose(M, np.eye(3)):  # identity
            return "Id()"

        else:  # actual rotations
            e, v = np.linalg.eig(M)

            if _naming_isclose(np.trace(M), -1.0):  # π-rotations
                angle = pi
                # rotation axis is eigenvector with eigenvalue +1
                axis = v[:, _naming_isclose(e, 1.0)].real.flatten()

            else:  # pick axis s.t. rotation angle be positive
                pos = e.imag > _naming_tol
                angle = np.angle(e[pos])[0]
                v = v[:, pos].flatten()
                axis = np.cross(v.imag, v.real)

            angle = int(np.rint(np.degrees(angle)))
            return f"Rot({angle}°){_to_int_vector(axis)}"

    elif _naming_isclose(np.linalg.det(M), -1.0):  # improper rotations

        if _naming_allclose(M, -np.eye(3)):  # inversion
            return "Inv()"

        elif _naming_isclose(np.trace(M), 1.0):  # reflections across a plane
            e, v = np.linalg.eig(M)
            # reflection plane normal is eigenvector with eigenvalue -1
            axis = v[:, _naming_isclose(e, -1.0)].real.flatten()
            return f"Refl{_to_int_vector(axis)}"

        else:  # rotoreflections, choose axis s.t. rotation angle be positive
            e, v = np.linalg.eig(M)
            pos = e.imag > _naming_tol
            angle = np.angle(e[pos])[0]
            angle = int(np.rint(np.degrees(angle)))
            v = v[:, pos].flatten()
            axis = np.cross(v.imag, v.real)
            return f"RotoRefl({angle}°){_to_int_vector(axis)}"

    else:
        raise ValueError("M must be an orthogonal matrix")


def __to_int_vector(v: Array) -> Array:
    # if there is a small integer representation, v/sum(abs(v)) consists of
    # rationals with small denominators
    v = v / np.abs(v).sum()
    scaled_v = np.outer(np.arange(1, 100), v)  # upper cutoff is arbitrary
    is_int_vector = np.all(_naming_isclose(scaled_v, np.rint(scaled_v)), axis=1)
    if np.any(is_int_vector):
        scaled_v = scaled_v[is_int_vector][0]
        return np.asarray(np.rint(scaled_v), dtype=int)
    else:
        raise ValueError


def _to_int_vector(v: Array) -> str:
    try:
        v = __to_int_vector(v)
        return f"[{v[0]},{v[1]},{v[2]}]"
    except ValueError:
        # in hexagonal symmetry, you often get a √3 in the x/y coordinate
        try:
            w = v.copy()
            w[1] /= 3 ** 0.5
            w = __to_int_vector(w)
            return f"[{w[0]},{w[1]}√3,{w[2]}]"
        except ValueError:
            # just return a normalised v
            v = v / np.linalg.norm(v)
            return f"[{v[0]:.3f},{v[1]:.3f},{v[2]:.3f}]"


############ POINT GROUP CLASS #################################################


@struct.dataclass()
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
            if isinstance(x, PGSymmetry) and (x.M.shape != (self.dim, self.dim)):
                raise ValueError(
                    "`PointGroup` contains operation of unexpected dimensionality"
                )

        # Define custom hash
        myhash = hash((super().__hash__(), hash(self.dim)))
        object.__setattr__(self, "_PointGroup__hash", myhash)

    def __matmul__(self, other) -> "PointGroup":
        if not isinstance(other, PointGroup):
            raise ValueError("Incompatible groups (`PointGroup` and something else)")

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

    @struct.property_cached
    def inverse(self) -> Array:
        lookup = self._canonical_lookup()

        inverse = np.zeros(len(self.elems), dtype=int)

        for index, element in enumerate(self.elems):
            # we exploit that the inverse of an orthogonal matrix is its transpose
            inverse_matrix = self._transformation_matrix(element).T
            inverse[index] = lookup[HashableArray(comparable(inverse_matrix))]

        return inverse

    @struct.property_cached
    def product_table(self) -> Array:
        # again, we calculate the product table of transformation matrices directly
        trans_matrices = self.to_array()
        product_matrices = np.einsum(
            "iab, jac -> ijbc", trans_matrices, trans_matrices
        )  # this is a table of M_g^{-1} M_h = M_g.T M_h
        product_matrices = comparable(product_matrices)

        lookup = self._canonical_lookup()

        n_symm = len(self)
        product_table = np.zeros((n_symm, n_symm), dtype=int)

        for i in range(n_symm):
            for j in range(n_symm):
                product_table[i, j] = lookup[HashableArray(product_matrices[i, j])]

        return product_table

    @property
    def shape(self) -> Shape:
        """Tuple `(<# of group elements>, <dim>, <dim>)`, same as :code:`self.to_array().shape`."""
        return (len(self), self.dim, self.dim)

    def __hash__(self):
        # pylint: disable=no-member
        return self.__hash
