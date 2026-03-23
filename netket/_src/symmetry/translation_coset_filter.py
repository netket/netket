# Copyright 2025 The NetKet Authors - All rights reserved.
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

from __future__ import annotations

from functools import cached_property

import numpy as np

from netket.operator import SumOperator, DiscreteJaxOperator
from netket._src.symmetry.translation_representation import TranslationRepresentation
from netket._src.symmetry.labeled_representation_coset_filter import (
    LabeledRepresentationCosetFilter,
)


def _cartesian_displacements(group, lattice) -> np.ndarray:
    """
    Cartesian displacement vectors (nearest-image) for all group elements.
    Shape: (|G|, ndim).
    """
    perms = np.asarray(group)
    positions = np.array(lattice.positions)
    basis = np.array(lattice.basis_vectors)
    extent = np.array(lattice.extent, dtype=float)

    d_cart = positions[0] - positions[perms[:, 0]]
    frac = d_cart @ np.linalg.inv(basis)
    frac -= np.round(frac / extent) * extent
    return frac @ basis


def _integer_displacements(group, lattice) -> np.ndarray:
    """
    Integer unit-cell displacements in [0, extent) for all group elements.
    Shape: (|G|, ndim).
    """
    perms = np.asarray(group)
    positions = np.array(lattice.positions)
    basis = np.array(lattice.basis_vectors)
    extent = np.array(lattice.extent, dtype=float)

    d_cart = positions[0] - positions[perms[:, 0]]
    frac = d_cart @ np.linalg.inv(basis)
    frac = frac % extent
    return np.round(frac).astype(int)


def _coset_rep_mask(full_group, sub_group, lattice) -> np.ndarray:
    """
    Boolean mask selecting coset reps of full_group / sub_group.

    A coset rep T^j satisfies 0 <= j_i < sub_stride_i for each active axis i.
    Returns bool array of shape (|G|,).
    """
    int_disps = _integer_displacements(full_group, lattice)

    sub_stride_full = np.zeros(lattice.ndim, dtype=int)
    for ax, s in zip(sub_group.axes, sub_group.strides):
        sub_stride_full[ax] = s

    active_axes = list(full_group.axes)
    if not active_axes:
        return np.ones(len(full_group.elems), dtype=bool)

    active_disps = int_disps[:, active_axes]
    active_sub_strides = sub_stride_full[active_axes]

    return np.all(
        (active_disps >= 0) & (active_disps < active_sub_strides[None, :]),
        axis=1,
    )


def _validate_subgroup(
    full_rep: TranslationRepresentation, sub_rep: TranslationRepresentation
) -> None:
    if full_rep.hilbert != sub_rep.hilbert:
        raise ValueError(
            "full_rep and sub_rep must have the same Hilbert space; "
            f"got {full_rep.hilbert!r} and {sub_rep.hilbert!r}."
        )

    full_lattice = full_rep.group.lattice
    sub_lattice = sub_rep.group.lattice
    if full_lattice is not sub_lattice and full_lattice != sub_lattice:
        raise ValueError("full_rep and sub_rep must have the same underlying lattice.")

    ndim = full_lattice.ndim
    full_stride_arr = np.zeros(ndim, dtype=int)
    for ax, s in zip(full_rep.group.axes, full_rep.group.strides):
        full_stride_arr[ax] = s

    sub_stride_arr = np.zeros(ndim, dtype=int)
    for ax, s in zip(sub_rep.group.axes, sub_rep.group.strides):
        sub_stride_arr[ax] = s

    for ax in full_rep.group.axes:
        fs = full_stride_arr[ax]
        ss = sub_stride_arr[ax]
        if ss == 0:
            raise ValueError(f"sub_rep does not cover axis {ax}, active in full_rep.")
        if ss % fs != 0:
            raise ValueError(
                f"sub_rep stride {ss} on axis {ax} must be a multiple of full_rep stride {fs}."
            )
        if ss < fs:
            raise ValueError(
                f"sub_rep stride {ss} on axis {ax} must be >= full_rep stride {fs}."
            )

    if all(sub_stride_arr[ax] == full_stride_arr[ax] for ax in full_rep.group.axes):
        raise ValueError(
            "sub_rep is identical to full_rep (same strides): "
            "the coset filter would have a single trivial element."
        )


class TranslationCosetFilter(LabeledRepresentationCosetFilter):
    """
    Fourier filter for the coset G/H of two translation representations.

    Subclass of :class:`~netket.symmetry.LabeledRepresentationCosetFilter`
    specialised for translation groups: characters are Bloch factors
    e^{-ik·d} rather than values from a character table.

    The full-group projector decomposes as::

        P_G(k) = projector_refinement(k) @ sub_rep.projector(k)

    Obtained via::

        C = full_rep.coset_filter(sub_rep)
    """

    def __init__(
        self,
        full_rep: TranslationRepresentation,
        sub_rep: TranslationRepresentation,
    ) -> None:
        _validate_subgroup(full_rep, sub_rep)
        self._full_rep = full_rep
        self._sub_rep = sub_rep

    # --- Translation-specific properties --------------------------------- #

    @property
    def lattice(self):
        """Underlying lattice (same for both reps)."""
        return self._full_rep.group.lattice

    @property
    def k_points(self) -> np.ndarray:
        """Full-group Bloch momenta (delegates to full_rep.k_points)."""
        return self._full_rep.k_points

    @property
    def irrep_labels(self) -> list[str]:
        """Full-group irrep labels (delegates to full_rep.irrep_labels)."""
        return self._full_rep.irrep_labels

    # Override to use a faster algorithm based on strides. Necessary for very big lattices
    @cached_property
    def _coset_mask(self) -> np.ndarray:
        return _coset_rep_mask(self._full_rep.group, self._sub_rep.group, self.lattice)

    @cached_property
    def _coset_displacements(self) -> np.ndarray:
        """Cartesian displacements of the coset reps. Shape (P, ndim)."""
        return _cartesian_displacements(self._full_rep.group, self.lattice)[
            self._coset_mask
        ]

    # override to use a faster algorithm based on displacements and Bloch factors
    def chars(self, k) -> np.ndarray:
        """
        Complex Bloch factors e^{-ik·d_c} for each coset representative c.

        Args:
            k: Bloch momentum — scalar (1D) or sequence (nD), one value per
               active axis of full_rep.

        Returns:
            Complex array of shape (P,).
        """
        active = self._full_rep.active_axes
        k_arr = np.asarray(k, dtype=float).ravel()
        k_full = np.zeros(self.lattice.ndim, dtype=float)
        k_full[list(active)] = k_arr
        return np.exp(-1j * self._coset_displacements @ k_full)

    # --- Override: projector_refinement supports both k= and label= ------ #

    def projector_refinement(
        self,
        k=None,
        *,
        label: str | None = None,
        atol: float = 1e-15,
    ) -> DiscreteJaxOperator:
        """
        Build the coset Fourier filter F_C(k).

        Exactly one of ``k`` or ``label`` must be supplied.

        Args:
            k:     Bloch momentum.
            label: Irrep label from :attr:`irrep_labels`.
            atol:  Drop terms with ``abs(coefficient)`` < atol.

        Raises:
            TypeError:  if neither or both of k and label are supplied.
            ValueError: if label is unknown.
        """
        if sum(x is not None for x in (k, label)) != 1:
            raise TypeError(
                "projector_refinement() requires exactly one of k= or label=."
            )

        if label is not None:
            try:
                idx = self.irrep_labels.index(label)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown label {label!r}. Valid: {self.irrep_labels}."
                ) from exc
            k = self.k_points[idx]

        k = np.asarray(k, dtype=float).ravel()
        raw_chars = self.chars(k)
        coefficients = np.conj(raw_chars) / self.n_coset_reps
        ops = np.array(self._coset_operators, dtype=object)
        keep = ~np.isclose(coefficients, 0.0, atol=atol)
        return SumOperator(*ops[keep], coefficients=coefficients[keep])

    def __repr__(self) -> str:
        full_g = self._full_rep.group
        sub_g = self._sub_rep.group
        return (
            f"TranslationCosetFilter("
            f"full_group={len(full_g.elems)} (strides={list(full_g.strides)}), "
            f"sub_group={len(sub_g.elems)} (strides={list(sub_g.strides)}), "
            f"coset_reps={self.n_coset_reps})"
        )
