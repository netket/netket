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

import numpy as np

from netket.operator import SumOperator


class LabeledRepresentationCosetFilter:
    """
    Coset refinement operator for G/H where G and H are finite groups.

    Used to extend the symmetry of a variational state one step at a time:
    if a state is already projected onto the H-symmetric sector (irrep ρ|_H),
    applying F_C(ρ) promotes it to the full G-symmetric sector (irrep ρ) using
    only ``len(G)/len(H)`` operator terms instead of the ``len(G)`` terms of the full projector.

    The refinement operator F_C(ρ) for irrep ρ of G satisfies::

        P_G(ρ) = F_C(ρ) @ P_H(ρ|_H)

    where C is a set of left-coset representatives of G/H.

    Obtained via::

        C = full_rep.coset_filter(sub_rep)
        F = C.projector_refinement(label="A1")  # |C| terms

    Chaining::

        C_d4_c2 = rep_d4.coset_filter(rep_c4).coset_filter(rep_c2)
    """

    def __init__(self, full_rep, sub_rep, *, _validate: bool = True) -> None:
        if _validate:
            # Check proper subgroup: types, same Hilbert space, and H < G.
            from netket._src.symmetry.labeled_representation import (
                LabeledRepresentation,
            )

            if not isinstance(full_rep, LabeledRepresentation):
                raise TypeError(
                    f"full_rep must be a LabeledRepresentation, got {type(full_rep)}"
                )
            if not isinstance(sub_rep, LabeledRepresentation):
                raise TypeError(
                    f"sub_rep must be a LabeledRepresentation, got {type(sub_rep)}"
                )
            if full_rep.hilbert != sub_rep.hilbert:
                raise ValueError(
                    "full_rep and sub_rep must share the same Hilbert space; "
                    f"got {full_rep.hilbert!r} and {sub_rep.hilbert!r}."
                )
            n_full = len(full_rep.group.elems)
            n_sub = len(sub_rep.group.elems)
            if not full_rep.group.is_subgroup(sub_rep.group, proper=True):
                raise ValueError(
                    "sub_rep must be a proper subgroup of full_rep: either the "
                    "groups are equal, |full| is not divisible by |sub|, or at "
                    "least one element of sub_rep is not in full_rep "
                    f"(|full|={n_full}, |sub|={n_sub})."
                )

        self._full_rep = full_rep
        self._sub_rep = sub_rep

    @property
    def full_rep(self):
        """The full-group representation G."""
        return self._full_rep

    @property
    def sub_rep(self):
        """The subgroup representation H."""
        return self._sub_rep

    @property
    def n_coset_reps(self) -> int:
        """``len(G) / len(H)`` — number of coset representatives."""
        return len(self._full_rep.group.elems) // len(self._sub_rep.group.elems)

    @property
    def _coset_mask(self) -> np.ndarray:
        """Bool mask of shape (|G|,) selecting one representative per left-coset.

        np.asarray(group) returns inverse-permutation arrays (shape (|G|, N));
        coset composition: (g*h)^{-1} = h^{-1}[g^{-1}] via numpy fancy indexing.
        """
        G_arr = np.asarray(self._full_rep.group)
        H_arr = np.asarray(self._sub_rep.group)
        covered: set[tuple] = set()
        mask = np.zeros(len(G_arr), dtype=bool)
        for i, g_inv in enumerate(G_arr):
            key = tuple(g_inv)
            if key in covered:
                continue
            mask[i] = True
            for h_inv in H_arr:
                covered.add(tuple(h_inv[g_inv]))
        return mask

    @property
    def _coset_operators(self) -> tuple:
        return tuple(
            op for op, keep in zip(self._full_rep.operators, self._coset_mask) if keep
        )

    def chars(self, label: str) -> np.ndarray:
        """
        Character values χ_ρ(c) for each coset representative c.

        Args:
            label: Irrep label from ``full_rep.irrep_labels``.

        Returns:
            Complex array of shape (P,).
        """
        idx = self._full_rep.irrep_labels.index(label)
        ct = self._full_rep.group.character_table()
        return ct[idx, self._coset_mask]

    def projector_refinement(
        self,
        *,
        label: str,
        atol: float = 1e-15,
    ) -> SumOperator:
        """
        Build the coset refinement operator F_C(ρ).

        This returns F_C alone (``len(G)/len(H)`` terms).  To obtain the full
        projector compose with the sub-group projector::

            F_C @ sub_rep.projector(label=sub_label)  ==  full_rep.projector(label)

        Args:
            label: Irrep label from ``full_rep.irrep_labels``.
            atol:  Drop operator terms with ``abs(coefficient)`` < atol.

        Raises:
            ValueError: if label is unknown.
        """
        try:
            idx = self._full_rep.irrep_labels.index(label)
        except ValueError as e:
            raise ValueError(
                f"Unknown label {label!r}. Valid: {self._full_rep.irrep_labels}."
            ) from e

        ct = self._full_rep.group.character_table()
        d_rho = round(ct[idx, 0].real)
        chars = ct[idx, self._coset_mask]

        coefficients = d_rho * np.conj(chars) / self.n_coset_reps
        keep = ~np.isclose(coefficients, 0.0, atol=atol)
        kept_ops = [op for op, k in zip(self._coset_operators, keep) if k]
        return SumOperator(*kept_ops, coefficients=coefficients[keep])

    def coset_filter(self, subgroup) -> LabeledRepresentationCosetFilter:
        """
        Build the next coset filter in a refinement chain.

        Delegates to :meth:`sub_rep.coset_filter(subgroup)
        <netket._src.symmetry.labeled_representation.LabeledRepresentation.coset_filter>`,
        so the result has ``full_rep = self.sub_rep`` and ``sub_rep = subgroup``::

            C = rep_d4.coset_filter(rep_c4).coset_filter(rep_c2)
            # equivalent to: rep_c4.coset_filter(rep_c2)

        See :meth:`LabeledRepresentation.coset_filter
        <netket._src.symmetry.labeled_representation.LabeledRepresentation.coset_filter>`
        for the full documentation.
        """
        return self._sub_rep.coset_filter(subgroup)

    def __repr__(self) -> str:
        n_full = len(self._full_rep.group.elems)
        n_sub = len(self._sub_rep.group.elems)
        return (
            f"{type(self).__name__}("
            f"size={self.n_coset_reps} ({n_full}/{n_sub}), "
            f"full_group={n_full} elements, "
            f"sub_group={n_sub} elements)"
        )
