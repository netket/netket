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
from typing import TYPE_CHECKING

import numpy as np

from netket.operator import SumOperator, DiscreteJaxOperator

from netket._src.symmetry.labeled_representation import (
    LabeledRepresentation,
    _label_to_character_index,
    _fmt_labels,
)

if TYPE_CHECKING:
    from netket._src.symmetry.translation_coset_filter import TranslationCosetFilter


def _fmt_k(v: float) -> str:
    """Format a k value as a compact multiple of π: '0', '0.5π', 'π', '-π', etc."""
    r = round(float(v) / np.pi, 6)
    if abs(r) < 1e-9:
        return "0"
    if abs(r - round(r)) < 1e-6:
        n = int(round(r))
        if n == 1:
            return "π"
        if n == -1:
            return "-π"
        return f"{n}π"
    return f"{r:.4g}π"


class TranslationRepresentation(LabeledRepresentation):
    """A :class:`~netket.symmetry.Representation` for the translation group of
    a :class:`~netket.graph.Lattice`, with Bloch-momentum indexing.

    Extends :class:`LabeledRepresentation` with:

    * :meth:`k_points` — ``(n_irreps, n_active_axes)`` array of Bloch momenta.
    * :attr:`irrep_labels` — ``"k=vπ"`` (1D) or ``"k=(v0π, v1π)"`` (nD) labels.
    * :meth:`projector` accepting a ``k=`` argument in addition
      to ``label=`` and the integer ``character_index``.

    The projection operator for a given k is built directly from
    :meth:`~netket.graph.space_group.TranslationGroup.momentum_irrep`, without
    consulting the character table.

    K-vector convention
        ``k`` is the Bloch momentum: ``k=π`` is the zone boundary, ``k=0`` is Γ.
        Valid values are :math:`k = 2\\pi m/L` for :math:`m = 0, 1, \\ldots, L-1`.
        Values lie in the half-open interval ``(-\\pi, \\pi]``.

    Examples
    --------
    ::

        lattice = nk.graph.Square(4, pbc=True)
        hi = nk.hilbert.Spin(0.5, lattice.n_nodes)

        rep = nk.symmetry.canonical_representation(hi, lattice.translation_group())

        print(rep.irrep_labels)
        # ['k=(0, 0)', 'k=(0.5π, 0)', 'k=(0, 0.5π)', ...]

        proj_gamma = rep.projector(k=(0.0, 0.0))
        proj_x     = rep.projector(k=(np.pi / 2, 0.0))
        proj_label = rep.projector(label="k=(0.5π, 0)")
        proj_index = rep.projector(0)              # backward compat

    """

    def __init__(self, group, representation_dict):
        """
        Args:
            group: A :class:`~netket.graph.space_group.TranslationGroup`.
            representation_dict: Mapping from group elements to operators.
        """
        from netket.graph.space_group import TranslationGroup

        if not isinstance(group, TranslationGroup):
            raise TypeError(
                f"group must be a TranslationGroup, got {type(group).__name__}"
            )
        super().__init__(group, representation_dict)

    @cached_property
    def active_axes(self) -> tuple[int, ...]:
        """Indices of lattice axes with more than one translation step."""
        shape = np.asarray(self.group.group_shape)
        return tuple(int(i) for i in np.where(shape > 1)[0])

    @property
    def k_points(self) -> np.ndarray:
        """Bloch momenta for all irreps.

        Returns an array of shape ``(n_irreps, n_active_axes)``.

        Values are consistent with :meth:`~netket.graph.space_group.TranslationGroup.momentum_irrep`:
        the m-th irrep corresponds to k such that ``to_reciprocal_lattice(k) == m`` (integer).

        For a strided group (``strides > 1``), the BZ boundary is at ``π / stride``,
        not at ``π``; this method accounts for that via the actual lattice dimensions.
        """
        shape = np.asarray(self.group.group_shape)
        active_idx = np.where(shape > 1)[0]
        active_shape = shape[active_idx]

        ranges = [np.arange(s) for s in active_shape]
        grids = np.meshgrid(*ranges, indexing="ij")
        m_all = np.stack([g.ravel() for g in grids], axis=1)  # (n_irreps, n_active)

        # Center the BZ: map m → m_centered ∈ (-ni/2, ni/2]
        m_centered = ((m_all + active_shape // 2) % active_shape) - active_shape // 2

        # Convert integer mode indices to Cartesian k using the actual lattice geometry.
        # to_reciprocal_lattice(k) = k @ _lattice_dims.T / (2π) must equal m (integer).
        # Inverting: k = 2π * m @ inv(_lattice_dims.T)
        ldims = np.asarray(self.group.lattice._lattice_dims, dtype=float)
        inv_ldims_T = np.linalg.inv(ldims.T)

        ndim = self.group.lattice.ndim
        m_full = np.zeros((len(m_all), ndim))
        m_full[:, active_idx] = m_centered.astype(float)

        k_cart = (2.0 * np.pi * m_full) @ inv_ldims_T  # (n_irreps, ndim)
        return k_cart[:, active_idx]

    @cached_property
    def irrep_labels(self) -> list[str]:
        """``"k=vπ"`` (1D) or ``"k=(v0π, v1π, ...)"`` (nD) labels for every irrep.

        Zero components are formatted as ``"0"`` (no π suffix).
        """
        kpts = self.k_points
        n_active = kpts.shape[1]
        if n_active == 1:
            return [f"k={_fmt_k(kpts[i, 0])}" for i in range(len(kpts))]
        return [
            "k=(" + ", ".join(_fmt_k(kpts[i, a]) for a in range(n_active)) + ")"
            for i in range(len(kpts))
        ]

    def projector(self, character_index=None, *, k=None, label=None, atol=1e-15) -> DiscreteJaxOperator:
        """Build the projection operator onto a momentum sector.

        Exactly one of ``character_index``, ``k``, or ``label`` must be provided.

        When ``k`` or ``label`` is given the projector is built directly from
        :meth:`~netket.graph.space_group.TranslationGroup.momentum_irrep`,
        bypassing the character table entirely. When only ``character_index``
        is given it falls back to the character-table path of the base class.

        Args:
            character_index: Integer index into the character table (0-based).
                Provided for backward compatibility with
                :class:`~netket.symmetry.Representation`.
            k: Bloch momentum as a scalar (1D) or sequence (nD), one component
                per active translation axis. ``k=0`` is the Γ point, ``k=π``
                is the zone boundary. Valid values are :math:`k = 2\\pi m / L`
                for integer :math:`m`.
            label: Irrep label string from :attr:`irrep_labels`,
                e.g. ``"k=0.5π"`` (1D) or ``"k=(0.5π, 0)"`` (2D).
            atol: Absolute tolerance for dropping near-zero projector terms.

        Returns:
            A :class:`~netket.operator.SumOperator` representing the projection
            operator :math:`P_k = \\frac{1}{|G|} \\sum_g \\chi_k(g)^* \\, T_g`
            onto the momentum-``k`` sector.

        Raises:
            TypeError: If not exactly one of the three selector arguments is given.
            InvalidWaveVectorError: If ``k`` is not a valid Brillouin-zone
                momentum for this lattice.
            ValueError: If ``label`` is not in :attr:`irrep_labels`.
        """
        n_given = sum(x is not None for x in (character_index, k, label))
        if n_given > 1:
            raise TypeError("Pass exactly one of character_index, k=, or label=.")
        if n_given == 0:
            raise TypeError(
                "projector() requires one of: character_index, k=, or label=."
            )

        if label is not None:
            idx = _label_to_character_index(self.irrep_labels, label)
            k = self.k_points[idx]

        # if w have a k wave-vector specified, use this fast path
        if k is not None:
            shape = np.asarray(self.group.group_shape)
            k_full = np.zeros(len(shape))
            k_full[shape > 1] = np.asarray(k, dtype=float).ravel()

            chars = self.group.momentum_irrep(*k_full)

            # Build a SumOperator projector from a character vector
            coefficients = np.conj(chars) / len(self.group.elems)
            ops = np.array(list(self.operators), dtype=object)
            mask = ~np.isclose(coefficients, 0.0, atol=atol)
            return SumOperator(*ops[mask], coefficients=coefficients[mask])

        # default fallback through character table indexing
        return LabeledRepresentation.projector(self, character_index, atol=atol)

    def coset_filter(
        self,
        subgroup: TranslationRepresentation,
    ) -> TranslationCosetFilter:
        """
        Build the coset Fourier filter for this group ``G`` modulo a subgroup ``H``.

        Returns a :class:`TranslationCosetFilter` storing the ``len(G)/len(H)``
        coset-representative operators.

        The filter satisfies::

            F_C(k) @ sub_rep.projector(k)  ==  full_rep.projector(k)

        for any full-lattice momentum k.

        Args:
            subgroup: :class:`TranslationRepresentation` for H <= G.
                Must have strides that are component-wise multiples of
                self's strides.

        Returns:
            :class:`TranslationCosetFilter` with ``len(G)/len(H)`` coset operators.

        Example — single level::

            lattice = nk.graph.Chain(8, pbc=True)
            hi      = nk.hilbert.Spin(0.5, 8)
            T_full  = nk.symmetry.canonical_representation(hi, lattice.translation_group())
            T_half  = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))

            C = T_full.coset_filter(T_half)
            # C has 2 coset representatives: {T^0, T^1}

            # 2-term refinement filter (apply on top of a T_half-symmetrised state):
            F = C.projector_refinement(k=np.pi/2)
            # F_C @ T_half.projector(k=np.pi/2) == T_full.projector(k=np.pi/2)
        """
        from netket._src.symmetry.translation_coset_filter import TranslationCosetFilter

        return TranslationCosetFilter(self, subgroup)

    def __str__(self) -> str:
        n = len(self.group.elems)
        return f"TranslationRepresentation(hilbert={self.hilbert!r}, lattice={self.group.lattice!s}, {n} elements, {n} k-points)"

    def __repr__(self) -> str:
        try:
            labels = self.irrep_labels
            lbl_str = f"k_points=[{_fmt_labels(labels)}] ({len(labels)} total)"
        except Exception:
            lbl_str = f"k_points=({len(self.group.elems)} total)"
        g = self.group
        n_elems = len(g.elems)
        group_str = f"axes={list(g.axes)}"
        if any(s > 1 for s in g.strides):
            group_str += f", strides={list(g.strides)}"
        return (
            f"TranslationRepresentation(\n"
            f"  hilbert={self.hilbert!r},\n"
            f"  lattice={g.lattice!s},\n"
            f"  translations: {group_str}, {n_elems} elements,\n"
            f"  {lbl_str}\n)"
        )
