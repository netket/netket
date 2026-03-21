"""
Tests for TranslationCosetFilter and TranslationRepresentation.coset_filter().

Mathematical identity tested throughout:
    C.projector_refinement(k) @ T_sub.projector(k)  ==  T_full.projector(k)
"""

import numpy as np
import pytest
import netket as nk
from netket._src.symmetry.translation_coset_filter import TranslationCosetFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _proj_to_dict(proj) -> dict:
    """
    Convert a SumOperator projector to {perm_tuple: coefficient} dict.
    Used for operator equality checks.
    """
    d = {}
    for op, c in zip(proj.operators, proj.coefficients):
        key = tuple(np.asarray(op).ravel().tolist())
        d[key] = d.get(key, 0.0) + c
    return {k: v for k, v in d.items() if abs(v) > 1e-12}


def _ops_equal(p, q, atol=1e-10) -> bool:
    """Check that two SumOperators represent the same operator."""
    dp = _proj_to_dict(p)
    dq = _proj_to_dict(q)
    if set(dp) != set(dq):
        return False
    return all(abs(dp[k] - dq[k]) < atol for k in dp)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        dict(L=8, strides_sub=2),
        dict(L=8, strides_sub=4),
        dict(L=16, strides_sub=2),
        dict(L=16, strides_sub=4),
    ]
)
def chain_1d(request):
    p = request.param
    L = p["L"]
    lattice = nk.graph.Chain(L, pbc=True)
    hi = nk.hilbert.Spin(0.5, L)
    T_full = nk.symmetry.canonical_representation(hi, lattice.translation_group())
    T_sub = nk.symmetry.canonical_representation(
        hi, lattice.translation_group(strides=p["strides_sub"])
    )
    return hi, lattice, T_full, T_sub, p["strides_sub"]


@pytest.fixture
def square_2d():
    lattice = nk.graph.Square(4, pbc=True)  # 4×4 = 16 sites
    hi = nk.hilbert.Spin(0.5, 16)
    T_full = nk.symmetry.canonical_representation(hi, lattice.translation_group())
    T_sub = nk.symmetry.canonical_representation(
        hi, lattice.translation_group(strides=(2, 2))
    )
    return hi, lattice, T_full, T_sub


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_factory_method(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        assert isinstance(C, TranslationCosetFilter)

    def test_direct_constructor(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = TranslationCosetFilter(T_full, T_sub)
        assert isinstance(C, TranslationCosetFilter)

    def test_n_coset_reps(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        L = lattice.n_nodes  # noqa: F841
        expected_P = stride  # |G|/|H| = L / (L/stride) = stride
        assert C.n_coset_reps == expected_P

    def test_n_coset_reps_2d(self, square_2d):
        hi, lattice, T_full, T_sub = square_2d
        C = T_full.coset_filter(T_sub)
        assert C.n_coset_reps == 4  # 2×2 patch

    def test_error_same_group(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        with pytest.raises(ValueError, match="identical to full_rep"):
            T_full.coset_filter(T_full)

    def test_error_reversed_order(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        with pytest.raises(ValueError):
            T_sub.coset_filter(T_full)  # sub cannot contain full

    def test_error_different_hilbert(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        # Same lattice, different Hilbert type — triggers the Hilbert check
        hi2 = nk.hilbert.Fock(n_max=2, N=lattice.n_nodes)
        T_other = nk.symmetry.canonical_representation(hi2, lattice.translation_group())
        with pytest.raises(ValueError, match="Hilbert"):
            T_full.coset_filter(T_other)


# ---------------------------------------------------------------------------
# perms
# ---------------------------------------------------------------------------


class TestNCosetReps:
    def test_n_coset_reps_shape(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        assert C.n_coset_reps == stride

    def test_n_coset_reps_2d(self, square_2d):
        hi, lattice, T_full, T_sub = square_2d
        C = T_full.coset_filter(T_sub)
        assert C.n_coset_reps == 4


# ---------------------------------------------------------------------------
# chars
# ---------------------------------------------------------------------------


class TestChars:
    def test_chars_shape(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        k = T_full.k_points[0]
        assert C.chars(k).shape == (C.n_coset_reps,)

    def test_chars_identity_is_one(self, chain_1d):
        """chi_k(T^0) = exp(-ik*0) = 1 for all k.  Identity is always index 0."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points:
            assert abs(C.chars(k)[0] - 1.0) < 1e-12

    def test_chars_k0_all_ones(self, chain_1d):
        """k=0 implies all characters are 1."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        k0_idx = np.argmin(np.abs(T_full.k_points))
        k0 = T_full.k_points[k0_idx]
        assert np.allclose(k0, 0.0)
        chars = C.chars(k0)
        assert np.allclose(chars, 1.0), "all characters at k=0 must be 1"

    def test_chars_unit_modulus(self, chain_1d):
        """Characters are on the unit circle."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points:
            chars = C.chars(k)
            assert np.allclose(
                np.abs(chars), 1.0
            ), f"chars must have unit modulus, got {np.abs(chars)}"


# ---------------------------------------------------------------------------
# projector_refinement: P-term filter
# ---------------------------------------------------------------------------


class TestProjectorRefinement:
    def test_n_terms(self, chain_1d):
        """F_C(k) has at most P terms."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points:
            F = C.projector_refinement(k=k)
            assert len(F.operators) <= C.n_coset_reps

    def test_refinement_times_ph_equals_full(self, chain_1d):
        """F_C(k) @ P_H(k) must equal T_full.projector(k)."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points:
            F_C = C.projector_refinement(k=k)
            P_H = T_sub.projector(k=k)
            combined = F_C @ P_H
            P_direct = T_full.projector(k=k)
            assert _ops_equal(combined, P_direct), f"F_C @ P_H != P_G at k={k}"

    def test_refinement_times_ph_equals_full_2d(self, square_2d):
        hi, lattice, T_full, T_sub = square_2d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points[:4]:
            F_C = C.projector_refinement(k=k)
            P_H = T_sub.projector(k=k)
            combined = F_C @ P_H
            P_direct = T_full.projector(k=k)
            assert _ops_equal(combined, P_direct), f"F_C @ P_H != P_G at k={k}"


# ---------------------------------------------------------------------------
# label= interface
# ---------------------------------------------------------------------------


class TestLabelInterface:
    def test_label_agrees_with_k(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for label, k in zip(C.irrep_labels, C.k_points):
            P_by_label = C.projector_refinement(label=label)
            P_by_k = C.projector_refinement(k=k)
            assert _ops_equal(
                P_by_label, P_by_k
            ), f"label={label!r} and k={k} give different projectors"

    def test_invalid_label_raises(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        with pytest.raises(ValueError, match="Unknown label"):
            C.projector_refinement(label="k=NotALabel")

    def test_no_argument_raises(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        with pytest.raises(TypeError):
            C.projector_refinement()

    def test_both_arguments_raises(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        with pytest.raises(TypeError):
            C.projector_refinement(k=0.0, label="k=0")


# ---------------------------------------------------------------------------
# Iterative / multi-level decomposition
# ---------------------------------------------------------------------------


class TestIterativeDecomposition:
    """
    P_G(k) = C_1.projector_refinement(k)
           @ C_2.projector_refinement(k)
           @ T_coarse.projector(k)

    for a three-level chain T_fine ⊇ T_medium ⊇ T_coarse.
    """

    @pytest.mark.parametrize(
        "L,s1,s2",
        [
            (8, 2, 4),
            (16, 2, 4),
            (16, 4, 8),
        ],
    )
    def test_three_level_1d(self, L, s1, s2):
        assert s2 % s1 == 0, "s2 must be a multiple of s1"
        lattice = nk.graph.Chain(L, pbc=True)
        hi = nk.hilbert.Spin(0.5, L)

        T_fine = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_medium = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=s1)
        )
        T_coarse = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=s2)
        )

        C_1 = T_fine.coset_filter(T_medium)
        C_2 = T_medium.coset_filter(T_coarse)

        for k in T_fine.k_points:
            P_direct = T_fine.projector(k=k)
            P_iter = (
                C_1.projector_refinement(k=k)
                @ C_2.projector_refinement(k=k)
                @ T_coarse.projector(k=k)
            )
            assert _ops_equal(
                P_direct, P_iter
            ), f"three-level decomposition failed at k={k} (L={L}, s1={s1}, s2={s2})"

    def test_coset_reps_count_product(self):
        """|C_1| * |C_2| * |T_coarse| == |T_full|"""
        L = 16
        lattice = nk.graph.Chain(L, pbc=True)
        hi = nk.hilbert.Spin(0.5, L)
        T_fine = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_medium = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=2)
        )
        T_coarse = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=4)
        )

        C_1 = T_fine.coset_filter(T_medium)
        C_2 = T_medium.coset_filter(T_coarse)

        assert C_1.n_coset_reps * C_2.n_coset_reps * len(T_coarse.group.elems) == len(
            T_fine.group.elems
        )


# ---------------------------------------------------------------------------
# NQS / wavefunction usage
# ---------------------------------------------------------------------------


class TestNQSWorkflow:
    def test_wavefunction_symmetrization_k0(self):
        """For k=0 all characters are 1."""
        L = 8
        lattice = nk.graph.Chain(L, pbc=True)
        hi = nk.hilbert.Spin(0.5, L)
        T_full = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_half = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=2)
        )
        C = T_full.coset_filter(T_half)

        k0 = np.zeros(1)
        chars = C.chars(k0)
        assert np.allclose(chars, 1.0)

    def test_wavefunction_symmetrization_characters_unit_modulus(self):
        """Characters must lie on the unit circle."""
        L = 8
        lattice = nk.graph.Chain(L, pbc=True)
        hi = nk.hilbert.Spin(0.5, L)
        T_full = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_half = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=2)
        )
        C = T_full.coset_filter(T_half)

        for k in T_full.k_points:
            chars = C.chars(k)
            assert np.allclose(
                np.abs(chars), 1.0
            ), f"chars at k={k} are not on the unit circle: {chars}"

    def test_perms_chars_progressive_symmetrization(self):
        """Shapes are correct for a 3-level setup; chars match expected values."""
        L = 16
        lattice = nk.graph.Chain(L, pbc=True)
        hi = nk.hilbert.Spin(0.5, L)

        T1 = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T2 = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=2)
        )
        T4 = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=4)
        )

        C_1_2 = T1.coset_filter(T2)
        C_2_4 = T2.coset_filter(T4)

        assert C_1_2.n_coset_reps == 2
        assert C_2_4.n_coset_reps == 2

        k = np.array([3 * np.pi / 8])
        assert C_1_2.chars(k).shape == (2,)
        assert C_2_4.chars(k).shape == (2,)

        # Level 1 chars at k=3π/8: e^{-i*3π/8 * {0,1}}
        expected_c12 = np.exp(-1j * 3 * np.pi / 8 * np.array([0.0, 1.0]))
        assert np.allclose(C_1_2.chars(k), expected_c12, atol=1e-12)

        # Level 2 chars at k=3π/8: e^{-i*3π/8 * {0,2}}
        expected_c24 = np.exp(-1j * 3 * np.pi / 8 * np.array([0.0, 2.0]))
        assert np.allclose(C_2_4.chars(k), expected_c24, atol=1e-12)
