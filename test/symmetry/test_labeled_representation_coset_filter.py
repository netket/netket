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

"""Tests for LabeledRepresentationCosetFilter.

Uses a 3×3 Square lattice (9 sites, 2^9=512-state Hilbert space) so that
dense-matrix comparisons are cheap.  D4 has 8 distinct permutations on this
lattice; C4 has 4 and C2 has 2.
"""

import numpy as np
import pytest

import netket as nk
from netket.utils.group import PermutationGroup
from netket._src.symmetry.labeled_representation_coset_filter import (
    LabeledRepresentationCosetFilter,
)


# ---------------------------------------------------------------------------
# Fixtures — module-scoped so the groups are built only once
# ---------------------------------------------------------------------------

L = 3  # 3×3 = 9 sites, Hilbert dim 2^9 = 512


@pytest.fixture(scope="module")
def lattice():
    return nk.graph.Square(L, pbc=True)


@pytest.fixture(scope="module")
def hi():
    return nk.hilbert.Spin(0.5, L * L)


@pytest.fixture(scope="module")
def point_group_reps(lattice, hi):
    """Build C2, C4, D4 representations on the 3×3 lattice."""
    pg = lattice.point_group()
    elems = list(pg.elems)

    c2_elems = [e for e in elems if str(e) in ("Id()", "Rot(180°)")]
    c4_elems = [
        e for e in elems if str(e) in ("Id()", "Rot(90°)", "Rot(180°)", "Rot(-90°)")
    ]

    rep_c2 = nk.symmetry.canonical_representation(
        hi, PermutationGroup(c2_elems, degree=L * L), warn=False
    )
    rep_c4 = nk.symmetry.canonical_representation(
        hi, PermutationGroup(c4_elems, degree=L * L), warn=False
    )
    rep_d4 = nk.symmetry.canonical_representation(hi, pg, warn=False)
    return rep_c2, rep_c4, rep_d4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proj_to_dict(proj) -> dict:
    """Convert SumOperator to {perm_tuple: coefficient} for cheap comparison."""
    d = {}
    for op, c in zip(proj.operators, proj.coefficients):
        key = tuple(np.asarray(op).ravel().tolist())
        d[key] = d.get(key, 0.0) + c
    return {k: v for k, v in d.items() if abs(v) > 1e-12}


def _sum_ops_equal(p, q, atol=1e-10) -> bool:
    """Check two SumOperators represent the same operator (dict-based, no dense mat)."""
    dp, dq = _proj_to_dict(p), _proj_to_dict(q)
    if set(dp) != set(dq):
        return False
    return all(abs(dp[k] - dq[k]) < atol for k in dp)


def _ops_close_dense(op1, op2, atol=1e-10) -> bool:
    """Dense-matrix comparison — only use for small Hilbert spaces."""
    return np.allclose(np.array(op1.to_dense()), np.array(op2.to_dense()), atol=atol)


# ---------------------------------------------------------------------------
# Tests: construction and error handling
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_d4_c4(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c4)
        assert isinstance(C, LabeledRepresentationCosetFilter)
        assert C.n_coset_reps == 2  # |D4|/|C4| = 8/4

    def test_d4_c2(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c2)
        assert C.n_coset_reps == 4  # |D4|/|C2| = 8/2

    def test_c4_c2(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_c4.coset_filter(rep_c2)
        assert C.n_coset_reps == 2  # |C4|/|C2| = 4/2

    def test_error_same_group(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        with pytest.raises(ValueError, match="proper subgroup"):
            rep_c4.coset_filter(rep_c4)

    def test_error_not_subgroup(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        with pytest.raises(ValueError):
            rep_c2.coset_filter(rep_c4)  # C4 is not a subgroup of C2

    def test_error_different_hilbert(self, point_group_reps, lattice):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        hi2 = nk.hilbert.Fock(n_max=2, N=L * L)
        pg = lattice.point_group()
        elems = list(pg.elems)
        c4_elems = [
            e for e in elems if str(e) in ("Id()", "Rot(90°)", "Rot(180°)", "Rot(-90°)")
        ]
        rep_c4_other = nk.symmetry.canonical_representation(
            hi2, PermutationGroup(c4_elems, degree=L * L), warn=False
        )
        with pytest.raises(ValueError, match="Hilbert"):
            rep_d4.coset_filter(rep_c4_other)


# ---------------------------------------------------------------------------
# Tests: perms
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: chars
# ---------------------------------------------------------------------------


class TestChars:
    def test_chars_shape(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c4)
        for label in rep_d4.irrep_labels:
            assert C.chars(label).shape == (C.n_coset_reps,)

    def test_chars_match_character_table(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c4)
        ct = rep_d4.group.character_table()
        for i, label in enumerate(rep_d4.irrep_labels):
            np.testing.assert_allclose(C.chars(label), ct[i, C._coset_mask], atol=1e-12)

    def test_identity_char_equals_dim(self, point_group_reps):
        """χ_ρ(Id) = d_ρ for each irrep.  Identity is always coset rep index 0."""
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c4)
        ct = rep_d4.group.character_table()
        for j, label in enumerate(rep_d4.irrep_labels):
            d_rho = round(ct[j, 0].real)
            assert abs(C.chars(label)[0] - d_rho) < 1e-10


# ---------------------------------------------------------------------------
# Tests: projector_refinement
# ---------------------------------------------------------------------------


class TestProjectorRefinement:
    def test_n_terms(self, point_group_reps):
        """F_C has at most P terms."""
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c4)
        for label in rep_d4.irrep_labels:
            ct = rep_d4.group.character_table()
            if round(ct[rep_d4.irrep_labels.index(label), 0].real) > 1:
                continue  # skip multi-dim irreps
            F = C.projector_refinement(label=label)
            assert len(F.operators) <= C.n_coset_reps

    def test_refinement_times_ph_equals_full_projector(self, point_group_reps):
        """F_C @ P_H == P_G for 1D irreps (dense comparison on 3×3 lattice)."""
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c4)
        ct = rep_d4.group.character_table()

        for i, label in enumerate(rep_d4.irrep_labels):
            if round(ct[i, 0].real) > 1:
                continue
            F_C = C.projector_refinement(label=label)
            # Find the matching sub-irrep by character comparison
            ct_sub = rep_c4.group.character_table()
            G_arr = np.asarray(rep_d4.group)
            H_arr = np.asarray(rep_c4.group)
            G_map = {tuple(row): j for j, row in enumerate(G_arr)}
            H_in_G = np.array([G_map[tuple(h)] for h in H_arr])
            chars_on_H = ct[i, H_in_G]
            n_H = len(rep_c4.group.elems)
            mults = np.array(
                [
                    np.dot(chars_on_H, np.conj(ct_sub[mu])).real / n_H
                    for mu in range(ct_sub.shape[0])
                ]
            )
            sub_idx = int(np.argmax(mults))
            P_H = rep_c4.projector(character_index=sub_idx)
            combined = F_C @ P_H
            P_direct = rep_d4.projector(label=label)
            assert _ops_close_dense(
                combined, P_direct
            ), f"F_C @ P_H ≠ P_G for {label!r}"


# ---------------------------------------------------------------------------
# Tests: iterative decomposition
# ---------------------------------------------------------------------------


class TestIterativeDecomposition:
    def test_d4_c4_c2_trivial_irrep(self, point_group_reps):
        """F_{D4/C4} @ F_{C4/C2} @ P_{C2}(trivial) == P_{D4}(trivial)."""
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C_d4_c4 = rep_d4.coset_filter(rep_c4)
        C_c4_c2 = rep_c4.coset_filter(rep_c2)

        label_d4 = rep_d4.irrep_labels[0]
        label_c4 = rep_c4.irrep_labels[0]
        label_c2 = rep_c2.irrep_labels[0]

        composed = (
            C_d4_c4.projector_refinement(label=label_d4)
            @ C_c4_c2.projector_refinement(label=label_c4)
            @ rep_c2.projector(label=label_c2)
        )
        P_d4 = rep_d4.projector(label=label_d4)
        assert _ops_close_dense(composed, P_d4)


# ---------------------------------------------------------------------------
# Tests: chaining interface
# ---------------------------------------------------------------------------


class TestChaining:
    def test_chain_equals_direct(self, point_group_reps):
        """rep_d4.coset_filter(rep_c4).coset_filter(rep_c2) delegates to rep_c4.coset_filter(rep_c2)."""
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C_direct = rep_c4.coset_filter(rep_c2)
        C_chained = rep_d4.coset_filter(rep_c4).coset_filter(rep_c2)
        assert isinstance(C_chained, LabeledRepresentationCosetFilter)
        assert C_direct.n_coset_reps == C_chained.n_coset_reps

    def test_chain_full_sub_are_correct(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c4).coset_filter(rep_c2)
        assert C.full_rep is rep_c4
        assert C.sub_rep is rep_c2

    def test_repr(self, point_group_reps):
        rep_c2, rep_c4, rep_d4 = point_group_reps
        C = rep_d4.coset_filter(rep_c4)
        r = repr(C)
        assert "LabeledRepresentationCosetFilter" in r
        assert "size=2" in r
