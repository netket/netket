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
from typing import Optional, Tuple

from .semigroup import Identity, Element
from .group import Group

from netket.utils import HashableArray, struct
from netket.utils.float import comparable, comparable_periodic, is_approx_int
from netket.utils.types import Array, DType, Shape

############ POINT GROUP SYMMETRY CLASS ########################################


@struct.dataclass
class PGSymmetry(Element):
    """
    An abstract group element object to geometrically describe point group symmetries.

    Contains two data fields: an orthogonal matrix `_W` and an optional translation
    vector `_w`. Vectors :math:`vec x` are mapped to :math:`W\vec x + \vec w`.
    """

    _W: Array
    """
    A 2D array specifying the transformation. It has to be orthogonal (i.e. it 
    must be real and satisfy :math:`W^T W = 1`)
    """

    _w: Optional[Array] = None
    """
    An optional vector specifying the translation associated with the point group
    transformation. 

    Defaults to the null vector, i.e., transformations that leave the origin in place.
    """

    def __call__(self, x):
        if self._w is not None:
            return np.tensordot(x, self._W.T, axes=1) + self._w
        else:
            return np.tensordot(x, self._W.T, axes=1)

    def preimage(self, x):
        if self._w is not None:
            return np.tensordot(x - self._w, self._W, axes=1)
        else:
            return np.tensordot(x, self._W, axes=1)

    def __hash__(self):
        if self._w is not None:
            return hash(
                (HashableArray(comparable(self._W)), HashableArray(comparable(self._w)))
            )
        else:
            return hash(HashableArray(comparable(self._W)))

    def __eq__(self, other):
        if isinstance(other, Permutation):
            if self._w is None and self._w is None:
                return HashableArray(comparable(self._W)) == HashableArray(
                    comparable(other._W)
                )
            elif self._w is not None and self._w is not None:
                return HashableArray(comparable(self._W)) == HashableArray(
                    comparable(other._W)
                ) and HashableArray(comparable(self._w)) == HashableArray(
                    comparable(other._w)
                )
            else:
                return False
        else:
            return False

    def change_origin(self, origin: Array) -> "PGSymmetry":
        """Returns a `PGSymmetry` representing a pure point-group transformation
        around `origin` with transformation matrix `self._W`."""
        return PGSymmetry(self._W, (np.eye(self.ndim) - self._W) @ np.asarray(origin))

    @struct.property_cached
    def _name(self) -> str:
        if self._W.shape == (2, 2):
            return _2D_name(self._W, self._w)
        elif self._W.shape == (3, 3):
            return _3D_name(self._W, self._w)
        else:
            return f"PGSymmetry({self._W}, {self._w or ''})"

    def __repr__(self):
        return self._name

    # TODO how to represent _w?
    def __array__(self, dtype: DType = None):
        return np.asarray(self._W, dtype)

    @property
    def ndim(self):
        return self._W.shape[0]

    @property
    def matrix(self):
        return self._W

    @property
    def translation(self):
        return self._w if self._w is not None else np.zeros(self._W.shape[0])

    @property
    def is_symmorphic(self):
        """Returns False if _w is defined."""
        return self._w is None

    @property
    def is_proper(self):
        return np.isclose(np.linalg.det(self._W), 1.0)


@dispatch
def product(p: PGSymmetry, q: PGSymmetry):
    if p.is_symmorphic and q.is_symmorphic:
        return PGSymmetry(p.matrix @ q.matrix)
    else:
        return PGSymmetry(p.matrix @ q.matrix, p.matrix @ q.translation + p.translation)


############ NAMING 2D AND 3D POINT GROUP SYMMETRIES ###########################

_naming_tol = 1e-6
_naming_allclose = partial(np.allclose, atol=_naming_tol, rtol=0.0)
_naming_isclose = partial(np.isclose, atol=_naming_tol, rtol=0.0)


def _origin_trans(W: Array, w: Array) -> Tuple[Array, Array]:
    """Decomposes a point group symmetry into a pure (improper) rotation around
    an origin and a translation along the axis/plane of the transformation.
    Returns the tuple (origin, translation)."""
    e, v = np.linalg.eig(np.eye(W.shape[0]) - W)
    # eigenvectors with eigenvalue 1 correspond to translations
    trans_v = v[:, _naming_isclose(e, 0.0)]
    trans = trans_v @ trans_v.T.conj() @ w
    # eigenvectors with other eigenvalues allow shifting the origin
    origin_v = v[:, np.logical_not(_naming_isclose(e, 0.0))]
    origin_e = np.diag(1 / e[np.logical_not(_naming_isclose(e, 0.0))])
    origin = origin_v @ origin_e @ origin_v.T.conj() @ w
    return origin.real, trans.real


def _2D_name(W: Array, w: Optional[Array]) -> str:
    if W.shape != (2, 2):
        raise ValueError("This function names 2D symmetries")

    if w is None:
        origin = trans = np.zeros(2)
    else:
        origin, trans = _origin_trans(W, w)

    origin = "" if _naming_allclose(origin, 0.0) else f"O{_to_rational_vector(origin)}"
    trans = None if _naming_allclose(trans, 0.0) else f"{_to_rational_vector(trans)}"

    if _naming_isclose(np.linalg.det(W), 1.0):  # rotations
        if _naming_allclose(W, np.eye(2)):  # identity / translation
            if trans is None:
                return "Id()"
            else:
                return f"Translation{trans}"
        else:
            angle = np.arctan2(W[1, 0], W[0, 0])
            # in practice, all rotations are by integer degrees
            angle = int(np.rint(np.degrees(angle)))
            return f"Rot({angle}°){origin}"

    elif _naming_isclose(np.linalg.det(W), -1.0):  # reflections / glides
        axis = np.arctan2(W[1, 0], W[0, 0]) / 2
        axis = int(np.rint(np.degrees(axis)))
        if trans is None:
            return f"Refl({axis}°){origin}"
        else:
            return f"Glide{trans}{origin}"

    else:
        raise ValueError("W must be an orthogonal matrix")


def _3D_name(W: Array, w: Optional[Array]) -> str:
    if W.shape != (3, 3):
        raise ValueError("This function names 3D symmetries")

    if w is None:
        origin = trans = np.zeros(3)
    else:
        origin, trans = _origin_trans(W, w)

    origin = "" if _naming_allclose(origin, 0.0) else f"O{_to_rational_vector(origin)}"

    if _naming_isclose(np.linalg.det(W), 1.0):  # rotations / screws
        if _naming_allclose(W, np.eye(3)):  # identity / translation
            if _naming_allclose(trans, 0.0):
                return "Id()"
            else:
                return f"Translation{_to_rational_vector(trans)}"

        else:  # actual rotations / screws
            e, v = np.linalg.eig(W)

            if _naming_isclose(np.trace(W), -1.0):  # π-rotations
                angle = pi
                # rotation axis is eigenvector with eigenvalue +1
                axis = v[:, _naming_isclose(e, 1.0)].real.flatten()

            else:  # pick axis s.t. rotation angle be positive
                pos = e.imag > _naming_tol
                angle = np.angle(e[pos])[0]
                v = v[:, pos].flatten()
                axis = np.cross(v.imag, v.real)
                if not _naming_allclose(trans, 0.0):
                    # screws may have negative angles if trans, axis are opposite
                    angle *= np.sign(axis @ trans)

            angle = int(np.rint(np.degrees(angle)))
            if _naming_allclose(trans, 0.0):
                return f"Rot({angle}°){_to_int_vector(axis)}"
            else:
                return f"Screw({angle}°){_to_rational_vector(trans)}"

    elif _naming_isclose(np.linalg.det(W), -1.0):  # improper rotations

        if _naming_allclose(W, -np.eye(3)):  # inversion
            return f"Inv(){origin}"

        elif _naming_isclose(np.trace(W), 1.0):  # reflections / glides
            e, v = np.linalg.eig(W)
            # reflection plane normal is eigenvector with eigenvalue -1
            axis = v[:, _naming_isclose(e, -1.0)].real.flatten()
            # convention: first nonzero entry is positive
            axis *= np.sign(axis[np.logical_not(_naming_isclose(axis, 0.0))][0])
            if _naming_allclose(trans, 0.0):
                return f"Refl{_to_int_vector(axis)}{origin}"
            else:
                return (
                    f"Glide{_to_rational_vector(trans)}ax{_to_int_vector(axis)}{origin}"
                )

        else:  # rotoreflections, choose axis s.t. rotation angle be positive
            e, v = np.linalg.eig(W)
            pos = e.imag > _naming_tol
            angle = np.angle(e[pos])[0]
            angle = int(np.rint(np.degrees(angle)))
            v = v[:, pos].flatten()
            axis = np.cross(v.imag, v.real)
            return f"RotoRefl({angle}°){_to_int_vector(axis)}{origin}"

    else:
        raise ValueError("W must be an orthogonal matrix")


def __to_rational(x: float) -> Tuple[int, int]:
    denom = is_approx_int(x * np.arange(1, 100), atol=_naming_tol)
    if not denom.any():
        raise ValueError
    denom = np.arange(1, 100)[denom][0]
    return int(np.rint(x * denom)), denom


def _to_rational(x: float) -> str:
    try:
        numer, denom = __to_rational(x)
        return f"{numer}/{denom}" if denom != 1 else f"{numer}"
    except ValueError:
        # in hexagonal symmetry, you often get a √3 in the x/y coordinate
        try:
            numer, denom = __to_rational(x / 3 ** 0.5)
            numer = "" if numer == 1 else ("-" if numer == -1 else numer)
            denom = "" if denom == 1 else f"/{denom}"
            return f"{numer}√3{denom}"
        except ValueError:
            # just return it as it is
            return f"{x:.3f}"


def _to_rational_vector(v: Array) -> Array:
    return "[" + ",".join(map(_to_rational, v)) + "]"


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

    ndim: int
    """Dimensionality of point group operations."""

    unit_cell: Optional[Array] = None
    """Lattice vectors of the unit cell on which the point group acts.
    It is used to match non-symmorphic symmetries modulo lattice translations."""

    def __hash__(self):
        return super().__hash__()

    @struct.property_cached
    def is_symmorphic(self) -> bool:
        return all(
            [
                (True if isinstance(elem, Identity) else elem.is_symmorphic)
                for elem in self.elems
            ]
        )

    def __matmul__(self, other) -> "PointGroup":
        if not isinstance(other, PointGroup):
            raise ValueError("Incompatible groups (`PointGroup` and something else)")

        # Check if dimensions match
        if self.ndim != other.ndim:
            raise ValueError("PointGroups of different dimensions cannot be multiplied")

        elems = super().__matmul__(other).elems

        # Cases for presence or absence of unit cells
        if (self.unit_cell is None) and (other.unit_cell is None):
            return PointGroup(elems, self.ndim)
        elif (self.unit_cell is None) != (other.unit_cell is None):
            uc = self.unit_cell if self.unit_cell is not None else other.unit_cell
            return PointGroup(elems, self.ndim, uc)
        else:
            if np.allclose(self.unit_cell, other.unit_cell):
                return PointGroup(elems, self.ndim, self.unit_cell)
            else:
                raise ValueError(
                    "PointGroups for different unit cells cannot be multiplied"
                )

    def _matrix(self, x: Element) -> Array:
        if isinstance(x, Identity):
            return np.eye(self.ndim, dtype=float)
        elif isinstance(x, PGSymmetry):
            return x.matrix
        else:
            raise ValueError(
                "`PointGroup` only supports `Identity` and `PGSymmetry` elements"
            )

    def _translation(self, x: Element) -> Array:
        if isinstance(x, Identity):
            return np.zeros(self.ndim, dtype=float)
        elif isinstance(x, PGSymmetry):
            return x.translation
        else:
            raise ValueError(
                "`PointGroup` only supports `Identity` and `PGSymmetry` elements"
            )

    def _canonical_from_w(self, W: Array, w: Array) -> Array:
        if self.is_symmorphic:
            return comparable(W)
        else:
            if self.unit_cell is None:
                return np.vstack((comparable(W), comparable(w)))
            else:
                return np.vstack(
                    (
                        comparable(W),
                        comparable_periodic(w @ np.linalg.inv(self.unit_cell)),
                    )
                )

    def _canonical_from_big_matrix(self, Ww: Array) -> Array:
        if self.is_symmorphic:
            return self._canonical_from_w(Ww, None)
        else:
            ndim = self.ndim
            return self._canonical_from_w(Ww[0:ndim, 0:ndim], Ww[0:ndim, ndim])

    def _canonical(self, x: Element) -> Array:
        return self._canonical_from_w(self._matrix(x), self._translation(x))

    def to_array(self) -> Array:
        """
        Convert the abstract group operations to an array of transformation matrices

        For symmorphic groups, `self.to_array()[i]` contains the transformation
        matrix of the `i`th group element.
        For nonsymmorphic groups, `self.to_array()[i]` is a (d+1)×(d+1) block
        matrix of the form [[W,w],[0,1]]: multiplying these matrices is
        equivalent to multiplying the symmetry operations.
        """
        if self.is_symmorphic:
            return np.asarray([self._matrix(x) for x in self.elems])
        else:
            return np.asarray(
                [
                    np.block(
                        [
                            [self._matrix(x), self._translation(x)[:, np.newaxis]],
                            [np.zeros(self.ndim), 1],
                        ]
                    )
                    for x in self.elems
                ]
            )

    def matrices(self) -> Array:
        return np.asarray([self._matrix(x) for x in self.elems])

    def translations(self) -> Array:
        return np.asarray([self._translation(x) for x in self.elems])

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

        pgroup = PointGroup(group.elems, self.ndim, unit_cell=self.unit_cell)

        if return_inverse:
            return pgroup, inverse
        else:
            return pgroup

    def rotation_group(self) -> "PointGroup":
        """
        Returns a new `PointGroup` that represents the subgroup of rotations
        (i.e. symmetries where the determinant of the transformation matrix is +1)
        in `self`
        """
        subgroup = []
        for i in self.elems:
            if isinstance(i, Identity):
                subgroup.append(i)
            elif i.is_proper():
                subgroup.append(i)
        return PointGroup(subgroup, self.ndim, unit_cell=self.unit_cell)

    def change_origin(self, origin: Array) -> "PointGroup":
        """
        Returns a new `PointGroup`, `out`, such that all elements of `out`
        describe pure point-group transformations around `origin` and `out[i]`
        has the same transformation matrix as `self[i]`.
        """
        out = []
        for elem in self.elems:
            if isinstance(elem, Identity):
                out.append(Identity())
            else:
                out.append(elem.change_origin(origin))
        return PointGroup(out, self.ndim, unit_cell=self.unit_cell)

    @struct.property_cached
    def inverse(self) -> Array:
        lookup = self._canonical_lookup()
        trans_matrices = self.to_array()

        inverse = np.zeros(len(self.elems), dtype=int)

        for index in range(len(self)):
            inverse_matrix = np.linalg.inv(trans_matrices[index])
            inverse[index] = lookup[
                HashableArray(self._canonical_from_big_matrix(inverse_matrix))
            ]

        return inverse

    @struct.property_cached
    def product_table(self) -> Array:
        # again, we calculate the product table of transformation matrices directly
        trans_matrices = self.to_array()
        product_matrices = np.einsum(
            "iab, jbc -> ijac", trans_matrices, trans_matrices
        )  # this is a table of M_g M_h

        lookup = self._canonical_lookup()

        n_symm = len(self)
        product_table = np.zeros((n_symm, n_symm), dtype=int)

        for i in range(n_symm):
            for j in range(n_symm):
                product_table[i, j] = lookup[
                    HashableArray(
                        self._canonical_from_big_matrix(product_matrices[i, j])
                    )
                ]

        return product_table[self.inverse]  # reshuffle rows to match specs

    @property
    def shape(self) -> Shape:
        """Tuple `(<# of group elements>, <ndim>, <ndim>)`, same as :code:`self.to_array().shape`."""
        if self.is_symmorphic:
            return (len(self), self.ndim, self.ndim)
        else:
            return (len(self), self.ndim + 1, self.ndim + 1)


def trivial_point_group(ndim: int) -> PointGroup:
    return PointGroup([Identity()], ndim=ndim)
