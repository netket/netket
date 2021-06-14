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
from functools import partial
from math import pi
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.linalg import schur

from netket.utils import HashableArray, struct
from netket.utils.float import comparable, comparable_periodic, is_approx_int
from netket.utils.types import Array, Shape
from netket.utils.dispatch import dispatch

from ._group import FiniteGroup
from ._semigroup import Element, Identity

############ POINT GROUP SYMMETRY CLASS ########################################


@struct.dataclass
class PGSymmetry(Element):
    """
    An abstract group element object to geometrically describe point group symmetries.

    Construction: `PGSymmetry(W,w)` with an orthogonal matrix `W` and an optional
    translation vector `w` returns an object that maps vectors :math:`\vec x` to
    :math:`W\vec x + \vec w`.

    Internally, the transformation is stored as the affine matrix [[W,w],[0,1]],
    such that matrix inverses and multiplication correspond to inverting and
    multiplying the transformation itself.
    """

    _affine: Array
    """
    A 2D array specifying the affine transformation. It has to be of the block
    structure [[W,w],[0,1]].
    """

    def __pre_init__(self, W: Array, w: Optional[Array] = None) -> Tuple[Tuple, Dict]:
        W = np.asarray(W)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("W must be a 2D square matrix")
        ndim = W.shape[0]
        if not np.allclose(W.T @ W, np.eye(ndim)):
            raise ValueError("W must be an orthogonal matrix")
        if w is None:
            w = np.zeros((ndim, 1))
        else:
            w = np.asarray(w).reshape(ndim, 1)
        return (np.block([[W, w], [np.zeros(ndim), 1]]),), {}

    @property
    def affine_matrix(self) -> Array:
        """
        Returns the (d+1) × (d+1) dimensional matrix representing the action of
        `self` as an affine transformation.
        """
        return self._affine

    @property
    def matrix(self) -> Array:
        """
        Returns the d×d dimensional matrix representing the rotational action
        of `self`.
        """
        return self._affine[:-1, :-1]

    @property
    def translation(self) -> Array:
        """Returns the translation vector associated with `self`."""
        return self._affine[:-1, -1]

    @property
    def ndim(self) -> int:
        """Returns the dimension of vectors `self` acts on."""
        return self._affine.shape[0] - 1

    @property
    def is_proper(self) -> bool:
        """Returns True if `self` is a proper rotation (det(`self.matrix`) is +1)."""
        return np.isclose(np.linalg.det(self.matrix), 1.0)

    @property
    def is_symmorphic(self) -> bool:
        """Returns True if `self` leaves the origin in place."""
        return np.allclose(self.translation, 0.0)

    def preimage(self, x):
        """
        Returns the preimage of `x` under the transformation
        (i.e., `self @ self.preimage(x) == x` up to numerical accuracy.
        """
        return np.tensordot(x - self.translation, self.matrix, axes=1)

    def k_action(self, x):
        """
        Returns the action of `self` on the input without the associated translation.
        This is how the symmetry acts on wave vectors.
        """
        return np.tensordot(x, self.matrix.T, axes=1)

    def __hash__(self):
        return hash(HashableArray(comparable(self._affine)))

    def __eq__(self, other):
        if isinstance(other, PGSymmetry):
            return HashableArray(comparable(self._affine)) == HashableArray(
                comparable(other._affine)
            )
        else:
            return False

    def change_origin(self, origin: Array) -> "PGSymmetry":
        """Returns a `PGSymmetry` representing a pure point-group transformation
        around `origin` with transformation matrix `self._W`."""
        return PGSymmetry(
            self.matrix, (np.eye(self.ndim) - self.matrix) @ np.asarray(origin)
        )

    @struct.property_cached
    def _name(self) -> str:
        if self.ndim == 2:
            return _2D_name(self.matrix, self.translation)
        elif self.ndim == 3:
            return _3D_name(self.matrix, self.translation)
        else:
            return f"PGSymmetry({self.matrix}, {self.translation})"

    def __repr__(self) -> str:
        return self._name  # noqa: E0306


@dispatch
def product(p: PGSymmetry, x: Array):  # noqa: F811
    return np.tensordot(x, p.matrix.T, axes=1) + p.translation


@dispatch
def product(p: PGSymmetry, q: PGSymmetry):  # noqa: F811
    return PGSymmetry(p.matrix @ q.matrix, p.matrix @ q.translation + p.translation)


############ NAMING 2D AND 3D POINT GROUP SYMMETRIES ###########################

_naming_tol = 1e-6
_naming_allclose = partial(np.allclose, atol=_naming_tol, rtol=0.0)
_naming_isclose = partial(np.isclose, atol=_naming_tol, rtol=0.0)


# use Schur decomposition for eigenvalues of orthogonal W matrices to ensure
# that eigenvectors are always orthogonal
def _eig(W):
    e, v = schur(W, "complex")
    return np.diag(e), v


def _origin_trans(W: Array, w: Array) -> Tuple[Array, Array]:
    """Decomposes a point group symmetry into a pure (improper) rotation around
    an origin and a translation along the axis/plane of the transformation.
    Returns the tuple (origin, translation)."""
    e, v = _eig(np.eye(W.shape[0]) - W)
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


def _3D_name(W: Array, w: Optional[Array] = None) -> str:
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
            e, v = _eig(W)

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
                return f"Rot({angle}°){_to_int_vector(axis)}{origin}"
            else:
                return f"Screw({angle}°){_to_rational_vector(trans)}{origin}"

    elif _naming_isclose(np.linalg.det(W), -1.0):  # improper rotations

        if _naming_allclose(W, -np.eye(3)):  # inversion
            return f"Inv(){origin}"

        elif _naming_isclose(np.trace(W), 1.0):  # reflections / glides
            e, v = _eig(W)
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
            e, v = _eig(W)
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
class PointGroup(FiniteGroup):
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, PointGroup):
            return False
        if (self.unit_cell is None) != (other.unit_cell is None):
            return False
        if not np.allclose(self.unit_cell, other.unit_cell):
            return False
        # for simplicity we also require that they be ordered the same way
        for i, elem in enumerate(self.elems):
            if elem != other.elems[i]:
                return False
        return True

    def matrix(self, x: Element) -> Array:
        if isinstance(x, Identity):
            return np.eye(self.ndim, dtype=float)
        elif isinstance(x, PGSymmetry):
            return x.matrix
        else:
            raise ValueError(
                "`PointGroup` only supports `Identity` and `PGSymmetry` elements"
            )

    def translation(self, x: Element) -> Array:
        if isinstance(x, Identity):
            return np.zeros(self.ndim, dtype=float)
        elif isinstance(x, PGSymmetry):
            return x.translation
        else:
            raise ValueError(
                "`PointGroup` only supports `Identity` and `PGSymmetry` elements"
            )

    def affine_matrix(self, x: Element) -> Array:
        if isinstance(x, Identity):
            return np.eye(self.ndim + 1, dtype=float)
        elif isinstance(x, PGSymmetry):
            return x.affine_matrix
        else:
            raise ValueError(
                "`PointGroup` only supports `Identity` and `PGSymmetry` elements"
            )

    def _canonical_from_affine_matrix(self, affine: Array) -> Array:
        if self.unit_cell is None:
            return comparable(affine)
        else:
            return np.vstack(
                (
                    comparable(affine[0 : self.ndim, 0 : self.ndim]),
                    comparable_periodic(
                        affine[0 : self.ndim, self.ndim] @ np.linalg.inv(self.unit_cell)
                    ),
                )
            )

    def _canonical(self, x: Element) -> Array:
        return self._canonical_from_affine_matrix(self.affine_matrix(x))

    def to_array(self) -> Array:
        """
        Convert the abstract group operations to an array of transformation matrices

        For symmorphic groups, `self.to_array()[i]` contains the transformation
        matrix of the `i`th group element.
        For nonsymmorphic groups, `self.to_array()[i]` is a (d+1)×(d+1) block
        matrix of the form [[W,w],[0,1]]: multiplying these matrices is
        equivalent to multiplying the symmetry operations.
        """
        return np.asarray([self.affine_matrix(x) for x in self.elems])

    def matrices(self) -> Array:
        return np.asarray([self.matrix(x) for x in self.elems])

    def translations(self) -> Array:
        return np.asarray([self.translation(x) for x in self.elems])

    def __array__(self, dtype=None) -> Array:
        return np.asarray(self.to_array(), dtype=dtype)

    def remove_duplicates(self, *, return_inverse=False) -> "PointGroup":
        """
        Returns a new :code:`PointGroup` with duplicate elements (that is, elements
        which represent identical transformations) removed.

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
            elif i.is_proper:
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
        try:
            lookup = self._canonical_lookup()
            affine_matrices = self.to_array()

            inverse = np.zeros(len(self.elems), dtype=int)

            for index in range(len(self)):
                inverse_matrix = np.linalg.inv(affine_matrices[index])
                inverse[index] = lookup[
                    HashableArray(self._canonical_from_affine_matrix(inverse_matrix))
                ]

            return inverse
        except KeyError as err:
            raise RuntimeError(
                "PointGroup does not contain the inverse of all elements"
            ) from err

    @struct.property_cached
    def product_table(self) -> Array:
        try:
            # again, we calculate the product table of transformation matrices directly
            affine_matrices = self.to_array()
            product_matrices = np.einsum(
                "iab, jbc -> ijac", affine_matrices, affine_matrices
            )  # this is a table of M_g M_h

            lookup = self._canonical_lookup()

            n_symm = len(self)
            product_table = np.zeros((n_symm, n_symm), dtype=int)

            for i in range(n_symm):
                for j in range(n_symm):
                    product_table[i, j] = lookup[
                        HashableArray(
                            self._canonical_from_affine_matrix(product_matrices[i, j])
                        )
                    ]

            return product_table[self.inverse]  # reshuffle rows to match specs
        except KeyError as err:
            raise RuntimeError("PointGroup is not closed under multiplication") from err

    @property
    def shape(self) -> Shape:
        """
        Tuple `(<# of group elements>, <ndim>, <ndim>)`.
        Equivalent to :code:`self.to_array().shape`.
        """
        return (len(self), self.ndim + 1, self.ndim + 1)


def trivial_point_group(ndim: int) -> PointGroup:
    return PointGroup([Identity()], ndim=ndim)


@dispatch
def product(A: PointGroup, B: PointGroup):  # noqa: F811
    if A.ndim != B.ndim:
        raise ValueError("Incompatible groups (`PointGroup`s of different dimension)")
    if A.unit_cell is None:
        unit_cell = B.unit_cell
    else:
        if B.unit_cell is not None and not np.allclose(A.unit_cell, B.unit_cell):
            raise ValueError(
                "Incompatible groups (`PointGroup`s with different unit cells)"
            )
        unit_cell = A.unit_cell
    return PointGroup(
        elems=[a @ b for a, b in itertools.product(A.elems, B.elems)],
        ndim=A.ndim,
        unit_cell=unit_cell,
    )
