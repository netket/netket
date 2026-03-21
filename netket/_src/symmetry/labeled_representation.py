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


from netket._src.symmetry.representation import Representation

if TYPE_CHECKING:
    from netket._src.symmetry.labeled_representation_coset_filter import (
        LabeledRepresentationCosetFilter,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fmt_char(c: complex) -> str:
    """Format a root-of-unity character value as a compact string.

    Examples: +1, -1, +i, -i, exp(+0.6667πi).
    """
    c = complex(c)
    r, im = round(c.real, 6), round(c.imag, 6)
    if abs(im) < 1e-6:  # real
        if abs(r - 1) < 1e-6:
            return "+1"
        if abs(r + 1) < 1e-6:
            return "-1"
        return f"{r:+.4g}"
    if abs(r) < 1e-6:  # pure imaginary
        if abs(im - 1) < 1e-6:
            return "+i"
        if abs(im + 1) < 1e-6:
            return "-i"
        return f"{im:+.4g}i"
    # general: express as exp(θπi)
    theta_over_pi = round(np.angle(c) / np.pi, 6)
    return f"exp({theta_over_pi:+.4g}πi)"


def _find_separating_elements(char_table: np.ndarray) -> np.ndarray:
    """Find a minimal set of group-element indices whose character values
    together uniquely identify every irrep row.

    The identity (column 0, all-ones) is skipped. Elements are added greedily
    until all n_irreps rows are distinct.

    Returns integer array of column indices.
    """
    n_irreps, n_elems = char_table.shape
    selected: list[int] = []
    current_distinct = 1  # nothing selected → all rows look the same

    def _rows_as_tuples(cols):
        if not cols:
            return [() for _ in range(n_irreps)]
        sub = np.round(char_table[:, cols], 8)
        return [
            tuple(zip(sub[m].real.tolist(), sub[m].imag.tolist()))
            for m in range(n_irreps)
        ]

    for j in range(1, n_elems):  # j=0 is identity, skip
        candidate = selected + [j]
        nd = len(set(_rows_as_tuples(candidate)))
        if nd > current_distinct:
            selected.append(j)
            current_distinct = nd
        if current_distinct == n_irreps:
            break

    return np.array(selected, dtype=int)


def _fmt_labels(labels: list[str], n: int = 6) -> str:
    """Format a label list, showing first 3 + last 3 if longer than n."""
    if len(labels) <= n:
        return ", ".join(labels)
    return ", ".join(labels[:3]) + ", ..., " + ", ".join(labels[-3:])


def _label_to_character_index(labels: list[str], label: str) -> int:
    try:
        return labels.index(label)
    except ValueError:
        raise ValueError(
            f"Unknown label {label!r}.\n" f"Available labels: {labels}."
        ) from None


# ---------------------------------------------------------------------------
# LabeledRepresentation
# ---------------------------------------------------------------------------


class LabeledRepresentation(Representation):
    """A :class:`~netket.symmetry.Representation` with automatic irrep labeling.

    Extends the base :class:`~netket.symmetry.Representation` with:

    * :attr:`irrep_labels` — a human-readable string for each irrep, derived
      from character values on a minimal set of "separating" group elements.
      For example: ``"+1"``, ``"-1"``, ``"+i"``, ``"exp(+0.3333πi)"``, or
      tuples thereof for multi-generator groups.
    * :meth:`projector` accepting a ``label=`` keyword in addition to the
      integer ``character_index``.

    The constructor signature is identical to
    :class:`~netket.symmetry.Representation`, so existing code that passes a
    ``(group, representation_dict)`` pair works unchanged.

    Subclasses can override :attr:`irrep_labels` to provide physics-specific
    names (e.g. ``"even"``/``"odd"`` for Z₂).

    Examples
    --------
    Project a state onto the antisymmetric sector of a Z₂ group::

        rep = nk.symmetry.spin_flip_representation(hilbert)
        print(rep.irrep_labels)     # ['+1', '-1']
        proj = rep.projector(label='-1')

    """

    @cached_property
    def irrep_labels(self) -> list[str]:
        """Human-readable label for each irrep (one per ``character_index``).

        Derived from character values on the minimal set of separating group
        elements:

        * Single separator: ``"+1"``, ``"-1"``, ``"+i"``, etc.
        * Multiple separators: ``"(+1, -1)"``, ``"(-1, +i)"``, etc.

        Subclasses override this with domain-specific labels.
        """
        ct = self.group.character_table()  # (n_irreps, |G|)
        sep = _find_separating_elements(ct)
        chars = ct[:, sep]  # (n_irreps, n_sep)

        if chars.shape[1] == 1:
            return [_fmt_char(chars[m, 0]) for m in range(len(chars))]
        return [
            "(" + ", ".join(_fmt_char(chars[m, j]) for j in range(chars.shape[1])) + ")"
            for m in range(len(chars))
        ]

    def projector(self, character_index=None, *, label=None, atol=1e-15):
        """Build the projection operator for an irrep.

        Accepts either:

        * ``character_index`` (int): index into the character table, as in the
          base :class:`~netket.symmetry.Representation`.
        * ``label`` (str): human-readable label from :attr:`irrep_labels`.

        Args:
            character_index: Irrep index (0-based).
            label: Irrep label string, e.g. ``"+1"`` or ``"-1"``.
            atol: Tolerance for dropping near-zero projector terms.

        Returns:
            A :class:`~netket.operator.SumOperator` projection operator.

        Raises:
            ValueError: if ``label`` is not in :attr:`irrep_labels`.
            TypeError: if neither or both arguments are supplied.
        """
        if character_index is not None and label is not None:
            raise TypeError("Pass either character_index or label=, not both.")
        if label is not None:
            character_index = _label_to_character_index(self.irrep_labels, label)
        if character_index is None:
            raise TypeError(
                "projector() requires either a positional character_index "
                "or the keyword argument label=."
            )
        return super().projector(character_index, atol=atol)

    def __str__(self) -> str:
        try:
            n = len(self.irrep_labels)
            n_part = f", {n} irreps"
        except Exception:
            n_part = ""
        n_elems = len(self.group.elems)
        group_str = f"{type(self.group).__name__}({n_elems} elements)"
        return f"{type(self).__name__}(hilbert={self.hilbert!r}, group={group_str}{n_part})"

    def coset_filter(
        self,
        subgroup: LabeledRepresentation,
    ) -> LabeledRepresentationCosetFilter:
        """
        Build the coset Fourier filter for this group G modulo subgroup H.

        Args:
            subgroup: :class:`LabeledRepresentation` for a subgroup H ≤ G.

        Returns:
            :class:`~netket.symmetry.LabeledRepresentationCosetFilter` for G/H.

        Example::

            rep_d4 = nk.symmetry.canonical_representation(hi, lattice.point_group())
            rep_c4 = nk.symmetry.canonical_representation(hi, c4_subgroup)
            C = rep_d4.coset_filter(rep_c4)
        """
        from netket._src.symmetry.labeled_representation_coset_filter import (
            LabeledRepresentationCosetFilter,
        )

        return LabeledRepresentationCosetFilter(self, subgroup)

    def __repr__(self) -> str:
        try:
            labels = self.irrep_labels
            lbl_part = f"\n  irreps=[{_fmt_labels(labels)}] ({len(labels)} total)"
        except Exception:
            lbl_part = ""
        n_elems = len(self.group.elems)
        return (
            f"{type(self).__name__}(\n"
            f"  hilbert={self.hilbert!r},\n"
            f"  group={type(self.group).__name__}({n_elems} elements)"
            f"{lbl_part}\n)"
        )
