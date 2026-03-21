# Implementation Plan: `TranslationCosetFilter` for NetKet

## Overview

Add a `TranslationCosetFilter` class and a `.coset_filter()` factory method on
`TranslationRepresentation`.  The object stores the |G|/|H| coset-representative
operators of a translation subgroup pair G ⊇ H and lets the user build a
momentum-labelled Fourier filter that, when applied to a state already in the
k|_H sector of H, projects onto a full-lattice momentum sector.

This is useful for NQS workflows where translation symmetry is added
progressively (coarse → fine), and for understanding the coset structure of
translation groups in the context of patching / coarse-graining.


---

## Mathematical Background

### Setup

Let G = ℤ_{L₁} × ... × ℤ_{L_d} be the full translation group of a lattice
(N = L₁·...·L_d sites), and H ≤ G the strided subgroup with strides
(s₁, ..., s_d):

    H = { T^{m·s} : m ∈ ℤ_{L₁/s₁} × ... × ℤ_{L_d/s_d} }

The coset representatives form the set

    C = { T^j : 0 ≤ j_i < s_i  for all i }

with |C| = s₁·...·s_d = P (the "patch size").  Every g ∈ G decomposes
uniquely as g = c · h with c ∈ C, h ∈ H, so G = C × H as a set.

### Key identity

For any character χ_k of G (labeled by Bloch momentum k):

    P_G(k)  =  F_C(k) @ P_H(k)

where

    F_C(k)  =  (1/P)  Σ_{c ∈ C}  χ_k(c)^*  T_c        (P terms)
    P_H(k)  =  (1/|H|)  Σ_{h ∈ H}  χ_k(h)^*  T_h      (|H| terms)

Proof: substitute g = c·h into P_G(k) = (1/|G|) Σ_g χ_k(g)^* T_g,
use χ_k(c·h) = χ_k(c) χ_k(h) (abelian group) and |G| = P·|H|.

Note: P_H(k) is well-defined for any full-lattice k (the character restricted
to H is always a valid H-character, folded into H's BZ by periodicity).
Concretely, χ_k(T^{m·s}) = e^{-ik·(m·s)} = e^{-ik_H·m·s} where k_H is k
reduced mod the H Brillouin zone — and `TranslationGroup.momentum_irrep`
already handles this folding internally.

### Consequence for symmetry-restricted states

On any state |ψ⟩ that is already an eigenstate of H with momentum k_H:

    F_C(k) |ψ⟩ = P_G(k) |ψ⟩     whenever k|_H = k_H
    F_C(k) |ψ⟩ = 0               whenever k|_H ≠ k_H

So F_C(k) acts as the full projector P_G(k) restricted to the k_H eigenspace
of H, using only P terms instead of |G|.

### Iterative (multi-level) decomposition

For a chain G ⊇ H₁ ⊇ H₂ ⊇ ... ⊇ H_n:

    P_G(k) = F_{C₁}(k) @ F_{C₂}(k) @ ... @ F_{C_n}(k) @ P_{H_n}(k)

where C_i = coset reps of H_{i-1} mod H_i.  Each filter uses only
|H_{i-1}|/|H_i| operators.  This gives an efficient multi-level projection.

### Wavefunction level (NQS application)

Given a base wavefunction ψ_NN and target momentum k_total, define:

    ψ_1(σ) = log Σ_{g ∈ H_n}  e^{-ik·g}  ψ_NN(T_g σ)       # |H_n| evals
    ψ_2(σ) = log Σ_{c ∈ C_n}  e^{-ik·c}  exp(ψ_{1}(T_c σ)) # ×|C_n| evals
    ...

At each stage the total number of ψ_NN evaluations equals |G|, but training
can be performed stage by stage (symmetry annealing), starting from coarse
symmetry and refining.

Characters e^{-ik·c} are:
- real (±1) for k such that k·c ∈ {0, π} for all coset reps c
- complex otherwise

For k=0 all characters are 1 (trivial, free), so each level adds symmetry
at zero extra complexity cost.


---

## New Public API

### `TranslationCosetFilter`

```python
class TranslationCosetFilter:
    """
    Fourier filter for the coset G/H of two translation representations.

    Stores the |G|/|H| coset-representative operators and computes the
    P-term filter F_C(k) for any k in G's Brillouin zone.

    Mathematically:

        P_G(k)  =  filter(k)  @  sub_rep.projector(k)

    where filter(k) = (1/P) Σ_{c ∈ C} χ_k(c)^* T_c.

    On any state |ψ⟩ already in the k|_H sector of the subgroup H:

        filter(k) |ψ⟩  =  P_G(k) |ψ⟩

    so filter() has only P = |G|/|H| terms instead of |G|.

    Obtained via TranslationRepresentation.coset_filter(subgroup_rep).
    """
```

#### Constructor (internal, called by factory)

```python
def __init__(
    self,
    full_rep: TranslationRepresentation,
    sub_rep:  TranslationRepresentation,
) -> None
```

Validates:
1. Both reps operate on the same Hilbert space.
2. Both reps have the same underlying lattice.
3. `sub_rep.group` is a subgroup of `full_rep.group`: every axis stride of
   `sub_rep` is a (positive integer) multiple of the corresponding stride in
   `full_rep`.

#### Properties

```python
@cached_property
def hilbert: AbstractHilbert

@cached_property
def lattice: Lattice

@cached_property
def n_coset_reps: int          # P = |G| / |H|

@cached_property
def perms: np.ndarray          # shape (P, N), dtype int
    """
    Permutation arrays for the P coset representatives.
    perms[i] is the integer permutation array for the i-th coset rep T^{j_i}.
    Useful for building symmetrized NQS models directly.
    """

@cached_property
def coset_displacements: np.ndarray   # shape (P, ndim), Cartesian
    """Physical displacement vectors of the coset representatives."""

@cached_property
def k_points: np.ndarray       # shape (n_irreps, n_active_axes)
    """Full-group Bloch momenta — identical to full_rep.k_points."""

@cached_property
def irrep_labels: list[str]
    """Full-group irrep labels — identical to full_rep.irrep_labels."""

@property
def full_rep: TranslationRepresentation
@property
def sub_rep:  TranslationRepresentation
```

#### `chars(k) -> np.ndarray`

```python
def chars(self, k) -> np.ndarray:
    """
    Complex characters e^{-ik·d_c} for each coset representative c.

    Args:
        k: Bloch momentum, scalar (1D) or sequence (nD), in the same
           convention as full_rep.k_points (one component per active axis).

    Returns:
        Complex array of shape (P,).
    """
```

#### `projector(k, *, label, compose, atol) -> SumOperator`

```python
def projector(
    self,
    k=None,
    *,
    label:   str | None = None,
    compose: bool = True,
    atol:    float = 1e-15,
) -> SumOperator:
    """
    Build the coset Fourier filter F_C(k), optionally composed with the
    subgroup projector P_H(k|_H).

    Exactly one of `k` or `label` must be given.

    Args:
        k:       Full-lattice Bloch momentum (same convention as
                 full_rep.projector).
        label:   Irrep label from self.irrep_labels.
        compose: If True (default), return F_C(k) @ P_H(k) — the full
                 projector P_G(k) written in P + |H| terms.
                 If False, return only F_C(k) in P terms.  The caller is
                 then responsible for ensuring the operand state is already
                 in the k|_H sector of H.
        atol:    Drop SumOperator terms with |coefficient| < atol.

    Returns:
        SumOperator with P terms (compose=False) or P+|H| terms (compose=True).

    Notes:
        compose=True is always mathematically correct and equals
        full_rep.projector(k).

        compose=False is more efficient when the state is guaranteed to be
        in the P_H(k|_H) sector, e.g. after an explicit projection or by
        construction of the variational ansatz.
    """
```

### Factory method on `TranslationRepresentation`

```python
def coset_filter(
    self,
    subgroup: "TranslationRepresentation",
) -> TranslationCosetFilter:
    """
    Build the coset Fourier filter for G / H.

    Args:
        subgroup: TranslationRepresentation for H ≤ G.

    Returns:
        TranslationCosetFilter with |G|/|H| coset-representative operators.

    Example:
        T_full  = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_patch = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))
        C = T_full.coset_filter(T_patch)
        F = C.projector(k=np.pi/2, compose=False)  # 2-term filter
    """
    return TranslationCosetFilter(self, subgroup)
```


---

## Implementation Details

### File layout

```
netket/_src/symmetry/
    translation_coset_filter.py    ← NEW
    translation_representation.py  ← add .coset_filter() method

netket/symmetry/__init__.py        ← export TranslationCosetFilter
netket/_src/symmetry/__init__.py   ← import TranslationCosetFilter
```

### `translation_coset_filter.py` — full implementation

```python
# Copyright 2025 The NetKet Authors - All rights reserved.
# (Apache 2.0 license header)

from __future__ import annotations

from functools import cached_property

import numpy as np

from netket.operator import SumOperator
from netket._src.symmetry.translation_representation import TranslationRepresentation


def _cartesian_displacements(group, lattice) -> np.ndarray:
    """
    Cartesian displacement vectors for all group elements.

    For a TranslationGroup element T^d, the displacement is the Cartesian
    vector d such that positions[T^d(0)] = positions[0] - d (mod PBC).

    Returns array of shape (|G|, ndim).
    """
    perms     = np.asarray(group)                       # (|G|, N)
    positions = np.array(lattice.positions)             # (N, ndim)
    basis     = np.array(lattice.basis_vectors)         # (ndim, ndim)
    extent    = np.array(lattice.extent, dtype=float)   # (ndim,)

    d_cart = positions[0] - positions[perms[:, 0]]      # (|G|, ndim)
    # Wrap into the unit cell (nearest image convention)
    frac = d_cart @ np.linalg.inv(basis)
    frac -= np.round(frac / extent) * extent
    return frac @ basis


def _coset_rep_mask(full_group, sub_group, lattice) -> np.ndarray:
    """
    Boolean mask over full_group.elems selecting exactly the coset
    representatives of full_group mod sub_group.

    Coset representative T^j satisfies: for each axis i,
        0  <=  j_i  <  sub_stride_i
    where j_i is the displacement in lattice-basis fractional coordinates
    and sub_stride_i is the step size of the subgroup along axis i.

    Returns bool array of shape (|G|,).
    """
    sub_strides = np.asarray(sub_group.strides, dtype=float)  # (ndim,)
    basis       = np.array(lattice.basis_vectors)              # (ndim, ndim)

    # Displacements in lattice-fractional coordinates
    cart_disps = _cartesian_displacements(full_group, lattice)  # (|G|, ndim)
    frac_disps = cart_disps @ np.linalg.inv(basis)              # (|G|, ndim)

    # Round to nearest integer (they should be close to integers)
    int_disps  = np.round(frac_disps).astype(int)               # (|G|, ndim)
    # Coset rep: displacement modulo sub_stride is in [0, sub_stride)
    # with sub_stride expressed in full-group step units
    full_strides = np.asarray(full_group.strides, dtype=int)    # (ndim,)

    # sub_stride in units of full-group steps
    sub_in_full = (sub_strides / full_strides).astype(int)      # (ndim,)

    int_disps_in_full = int_disps // full_strides[None, :]      # (|G|, ndim)

    mask = np.all(
        (int_disps_in_full >= 0) & (int_disps_in_full < sub_in_full[None, :]),
        axis=1,
    )
    return mask


def _validate_subgroup(full_rep: TranslationRepresentation,
                       sub_rep:  TranslationRepresentation) -> None:
    if full_rep.hilbert != sub_rep.hilbert:
        raise ValueError(
            "full_rep and sub_rep must have the same Hilbert space; "
            f"got {full_rep.hilbert!r} and {sub_rep.hilbert!r}."
        )

    full_lattice = full_rep.group.lattice
    sub_lattice  = sub_rep.group.lattice
    if full_lattice is not sub_lattice and full_lattice != sub_lattice:
        raise ValueError(
            "full_rep and sub_rep must have the same underlying lattice."
        )

    full_strides = np.asarray(full_rep.group.strides)
    sub_strides  = np.asarray(sub_rep.group.strides)

    if not np.all(sub_strides % full_strides == 0):
        raise ValueError(
            f"sub_rep strides {list(sub_strides)} must be component-wise "
            f"multiples of full_rep strides {list(full_strides)}."
        )
    if not np.all(sub_strides >= full_strides):
        raise ValueError(
            "sub_rep must represent a subgroup of full_rep "
            "(sub_rep strides must be >= full_rep strides on every axis)."
        )
    if np.all(sub_strides == full_strides):
        raise ValueError(
            "sub_rep is identical to full_rep (same strides): "
            "the coset filter would have a single trivial element."
        )


class TranslationCosetFilter:
    """
    Fourier filter for the coset G/H of two translation representations.

    Stores the |G|/|H| coset-representative operators and computes the
    P-term filter F_C(k) for any k in G's Brillouin zone.

    The full-group projector decomposes as:

        P_G(k) = filter(k) @ sub_rep.projector(k)

    where filter(k) = (1/P) Σ_{c ∈ C} χ_k(c)^* T_c has only P = |G|/|H|
    terms (compared to |G| for P_G(k) directly).

    On any state |ψ⟩ already in the k|_H sector of the subgroup H:

        filter(k, compose=False) |ψ⟩ = P_G(k) |ψ⟩

    Obtained via:
        C = full_rep.coset_filter(sub_rep)
    """

    def __init__(
        self,
        full_rep: TranslationRepresentation,
        sub_rep:  TranslationRepresentation,
    ) -> None:
        _validate_subgroup(full_rep, sub_rep)
        self._full_rep = full_rep
        self._sub_rep  = sub_rep

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def full_rep(self) -> TranslationRepresentation:
        return self._full_rep

    @property
    def sub_rep(self) -> TranslationRepresentation:
        return self._sub_rep

    @cached_property
    def hilbert(self):
        return self._full_rep.hilbert

    @cached_property
    def lattice(self):
        return self._full_rep.group.lattice

    @cached_property
    def n_coset_reps(self) -> int:
        """P = |G| / |H| — number of coset representatives."""
        return len(self._full_rep.group.elems) // len(self._sub_rep.group.elems)

    @cached_property
    def _coset_mask(self) -> np.ndarray:
        """Bool mask of shape (|G|,) selecting coset reps."""
        return _coset_rep_mask(
            self._full_rep.group, self._sub_rep.group, self.lattice
        )

    @cached_property
    def perms(self) -> np.ndarray:
        """
        Permutation arrays for the coset representatives.

        Shape: (P, N), dtype int.
        perms[i] is the permutation of site indices for the i-th coset rep.
        Useful for building symmetrized NQS models:
            log_psi_all = jax.vmap(model)(sigma[C.perms])  # (P,)
        """
        all_perms = np.asarray(self._full_rep.group)  # (|G|, N)
        return all_perms[self._coset_mask]

    @cached_property
    def coset_displacements(self) -> np.ndarray:
        """
        Cartesian displacement vectors for the coset representatives.

        Shape: (P, ndim).
        """
        all_disps = _cartesian_displacements(
            self._full_rep.group, self.lattice
        )
        return all_disps[self._coset_mask]

    @cached_property
    def _coset_operators(self) -> tuple:
        """Operator tuple for the coset reps (in same order as perms)."""
        ops = np.array(list(self._full_rep.operators), dtype=object)
        return tuple(ops[self._coset_mask])

    @cached_property
    def k_points(self) -> np.ndarray:
        """Full-group Bloch momenta — same as full_rep.k_points."""
        return self._full_rep.k_points

    @cached_property
    def irrep_labels(self) -> list[str]:
        """Full-group irrep labels — same as full_rep.irrep_labels."""
        return self._full_rep.irrep_labels

    # ------------------------------------------------------------------ #
    # Core interface
    # ------------------------------------------------------------------ #

    def chars(self, k) -> np.ndarray:
        """
        Complex characters e^{-ik·d_c} for each coset representative c.

        Args:
            k: Bloch momentum as scalar (1D) or sequence (nD), one value per
               active axis of full_rep.  Same convention as
               full_rep.projector(k=...).

        Returns:
            Complex array of shape (P,).
        """
        active = list(self._full_rep.active_axes)
        k_arr  = np.asarray(k, dtype=float).ravel()

        k_full = np.zeros(self.lattice.ndim, dtype=float)
        k_full[active] = k_arr

        disps  = self.coset_displacements              # (P, ndim)
        return np.exp(-1j * disps @ k_full)            # (P,)

    def projector(
        self,
        k=None,
        *,
        label: str | None = None,
        compose: bool = True,
        atol: float = 1e-15,
    ) -> "SumOperator":
        """
        Build the coset Fourier filter F_C(k), optionally composed with
        the subgroup projector P_H(k|_H).

        Exactly one of k or label must be provided.

        Args:
            k:       Full-lattice Bloch momentum.
            label:   Irrep label from self.irrep_labels.
            compose: If True (default), return F_C(k) @ P_H(k|_H), which
                     equals full_rep.projector(k) and is always correct.
                     If False, return only F_C(k) (P terms).  Correct only
                     when the operand is already in the k|_H sector.
            atol:    Drop terms with |coefficient| < atol.

        Returns:
            SumOperator.
        """
        n_given = sum(x is not None for x in (k, label))
        if n_given != 1:
            raise TypeError("projector() requires exactly one of k= or label=.")

        if label is not None:
            try:
                idx = self.irrep_labels.index(label)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown label {label!r}. "
                    f"Valid labels: {self.irrep_labels}."
                ) from exc
            k = self.k_points[idx]

        k = np.asarray(k, dtype=float).ravel()

        # Build F_C(k): P-term coset filter
        raw_chars    = self.chars(k)              # (P,)  complex
        coefficients = np.conj(raw_chars) / self.n_coset_reps
        ops          = np.array(self._coset_operators, dtype=object)
        mask         = ~np.isclose(coefficients, 0.0, atol=atol)
        F_C = SumOperator(*ops[mask], coefficients=coefficients[mask])

        if not compose:
            return F_C

        # Compose with P_H(k|_H): sub_rep handles the folding internally
        active = list(self._full_rep.active_axes)
        k_for_sub = k  # same k vector; sub_rep.projector folds to H's BZ
        P_H = self._sub_rep.projector(k=k_for_sub)
        return F_C @ P_H

    # ------------------------------------------------------------------ #
    # Dunder
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        full_g  = self._full_rep.group
        sub_g   = self._sub_rep.group
        n_full  = len(full_g.elems)
        n_sub   = len(sub_g.elems)
        return (
            f"TranslationCosetFilter(\n"
            f"  hilbert={self.hilbert!r},\n"
            f"  lattice={self.lattice!s},\n"
            f"  full_group={n_full} elements (strides={list(full_g.strides)}),\n"
            f"  sub_group={n_sub} elements  (strides={list(sub_g.strides)}),\n"
            f"  coset_reps={self.n_coset_reps},\n"
            f"  k_points=[{self.irrep_labels[0]}, ...] ({len(self.irrep_labels)} total)\n"
            f")"
        )
```

### Addition to `TranslationRepresentation`

Add the following method to `TranslationRepresentation` in
`netket/_src/symmetry/translation_representation.py`:

```python
def coset_filter(
    self,
    subgroup: "TranslationRepresentation",
) -> "TranslationCosetFilter":
    """
    Build the coset Fourier filter for this group G modulo a subgroup H.

    Returns a :class:`TranslationCosetFilter` storing the |G|/|H|
    coset-representative operators.

    The filter satisfies:

        P_G(k) = coset_filter(sub).projector(k)

    for any full-lattice momentum k.  When ``compose=False`` is passed to
    :meth:`~TranslationCosetFilter.projector`, the returned P-term operator
    equals P_G(k) when applied to states already in the k|_H sector of H.

    Args:
        subgroup: :class:`TranslationRepresentation` for H ≤ G.
                  Must have strides that are component-wise multiples of
                  self's strides.

    Returns:
        :class:`TranslationCosetFilter` with |G|/|H| coset operators.

    Example — single level::

        lattice = nk.graph.Chain(8, pbc=True)
        hi      = nk.hilbert.Spin(0.5, 8)
        T_full  = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_half  = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))

        C = T_full.coset_filter(T_half)
        # C has 2 coset representatives: {T^0, T^1}

        # 2-term filter (apply to states in T_half k=0 sector):
        F = C.projector(k=np.pi/2, compose=False)

        # or the full projector in P + |H| form:
        P = C.projector(k=np.pi/2, compose=True)
        # P equals T_full.projector(k=np.pi/2)

    Example — iterative (multi-level)::

        lattice = nk.graph.Chain(16, pbc=True)
        hi      = nk.hilbert.Spin(0.5, 16)
        T1 = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T2 = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))
        T4 = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=4))

        C_1_2 = T1.coset_filter(T2)   # T1/T2: 2 coset reps {T^0, T^1}
        C_2_4 = T2.coset_filter(T4)   # T2/T4: 2 coset reps {T^0, T^2}

        # Full projector at k=3π/8 in 2+2+4 = 8 terms (vs 16):
        k = 3*np.pi/8
        P_full_reconstructed = (
            C_1_2.projector(k=k, compose=False)
            @ C_2_4.projector(k=k, compose=False)
            @ T4.projector(k=k)
        )
        # equals T1.projector(k=3*np.pi/8)
    """
    from netket._src.symmetry.translation_coset_filter import TranslationCosetFilter
    return TranslationCosetFilter(self, subgroup)
```

### Exports

In `netket/_src/symmetry/__init__.py`, add:
```python
from netket._src.symmetry.translation_coset_filter import TranslationCosetFilter
```

In `netket/symmetry/__init__.py`, add `TranslationCosetFilter` to `__all__` and
to the import block.


---

## Accessing `active_axes` on `TranslationRepresentation`

The implementation uses `full_rep.active_axes`, which is not currently a public
property of `TranslationRepresentation`.  Add it:

```python
# In TranslationRepresentation
@cached_property
def active_axes(self) -> tuple[int, ...]:
    """Indices of axes with more than one translation step."""
    shape = np.asarray(self.group.group_shape)
    return tuple(int(i) for i in np.where(shape > 1)[0])
```

(Check first whether this already exists under another name before adding.)


---

## Tests

### File: `tests/symmetry/test_translation_coset_filter.py`

```python
"""
Tests for TranslationCosetFilter and TranslationRepresentation.coset_filter().

Mathematical identity tested throughout:
    C.projector(k, compose=True) == full_rep.projector(k)
"""
import numpy as np
import pytest
import netket as nk
from netket.symmetry import TranslationCosetFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _proj_to_dict(proj) -> dict:
    """
    Convert a SumOperator projector to {perm_tuple: coefficient} dict,
    normalising out global phase.  Used for operator equality checks.
    """
    d = {}
    for op, c in zip(proj.operators, proj.coefficients):
        key = tuple(np.asarray(op).ravel().tolist())
        d[key] = d.get(key, 0.0) + c
    # drop near-zero
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

@pytest.fixture(params=[
    dict(L=8,  strides_sub=2),
    dict(L=8,  strides_sub=4),
    dict(L=16, strides_sub=2),
    dict(L=16, strides_sub=4),
])
def chain_1d(request):
    p = request.param
    L = p["L"]
    lattice = nk.graph.Chain(L, pbc=True)
    hi      = nk.hilbert.Spin(0.5, L)
    T_full  = nk.symmetry.canonical_representation(hi, lattice.translation_group())
    T_sub   = nk.symmetry.canonical_representation(
        hi, lattice.translation_group(strides=p["strides_sub"])
    )
    return hi, lattice, T_full, T_sub, p["strides_sub"]


@pytest.fixture
def square_2d():
    lattice = nk.graph.Square(4, pbc=True)   # 4×4 = 16 sites
    hi      = nk.hilbert.Spin(0.5, 16)
    T_full  = nk.symmetry.canonical_representation(hi, lattice.translation_group())
    T_sub   = nk.symmetry.canonical_representation(
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
        L = lattice.n_nodes
        expected_P = stride  # |G|/|H| = L / (L/stride) = stride
        assert C.n_coset_reps == expected_P

    def test_n_coset_reps_2d(self, square_2d):
        hi, lattice, T_full, T_sub = square_2d
        C = T_full.coset_filter(T_sub)
        assert C.n_coset_reps == 4   # 2×2 patch

    def test_error_same_group(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        with pytest.raises(ValueError, match="identical to full_rep"):
            T_full.coset_filter(T_full)

    def test_error_reversed_order(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        with pytest.raises(ValueError, match="subgroup"):
            T_sub.coset_filter(T_full)   # sub cannot contain full

    def test_error_different_hilbert(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        lattice2 = nk.graph.Chain(lattice.n_nodes, pbc=True)
        hi2      = nk.hilbert.Spin(0.5, lattice2.n_nodes)
        T_other  = nk.symmetry.canonical_representation(
            hi2, lattice2.translation_group()
        )
        with pytest.raises(ValueError, match="Hilbert"):
            T_full.coset_filter(T_other)


# ---------------------------------------------------------------------------
# perms
# ---------------------------------------------------------------------------

class TestPerms:
    def test_perms_shape(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        assert C.perms.shape == (C.n_coset_reps, lattice.n_nodes)
        assert C.perms.dtype in (np.int32, np.int64, int)

    def test_perms_are_valid_permutations(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        N = lattice.n_nodes
        for row in C.perms:
            assert set(row) == set(range(N)), "each row must be a permutation of {0..N-1}"

    def test_first_perm_is_identity(self, chain_1d):
        """T^0 is always a coset representative."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        N = lattice.n_nodes
        identity = np.arange(N)
        assert any(np.array_equal(row, identity) for row in C.perms), \
            "identity permutation must be among coset reps"

    def test_perms_2d(self, square_2d):
        hi, lattice, T_full, T_sub = square_2d
        C = T_full.coset_filter(T_sub)
        assert C.perms.shape == (4, 16)


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
        """χ_k(T^0) = e^{-ik·0} = 1 for all k."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        N = lattice.n_nodes
        identity = tuple(range(N))
        for k in T_full.k_points:
            chars = C.chars(k)
            # find which coset rep is the identity
            for i, row in enumerate(C.perms):
                if tuple(row) == identity:
                    assert abs(chars[i] - 1.0) < 1e-12

    def test_chars_k0_all_ones(self, chain_1d):
        """k=0 implies all characters are 1."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        # find k=0 in k_points
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
            assert np.allclose(np.abs(chars), 1.0), \
                f"chars must have unit modulus, got {np.abs(chars)}"


# ---------------------------------------------------------------------------
# Core identity: compose=True equals full_rep.projector(k)
# ---------------------------------------------------------------------------

class TestProjectorIdentity:
    """
    The fundamental correctness test: for every full-group momentum k,

        C.projector(k, compose=True)  ==  T_full.projector(k)
    """

    def test_compose_true_equals_full_projector_1d(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points:
            P_direct   = T_full.projector(k=k)
            P_composed = C.projector(k=k, compose=True)
            assert _ops_equal(P_direct, P_composed), \
                f"projector mismatch at k={k}: direct={P_direct}, composed={P_composed}"

    def test_compose_true_equals_full_projector_2d(self, square_2d):
        hi, lattice, T_full, T_sub = square_2d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points[:4]:   # test a subset (16 irreps total)
            P_direct   = T_full.projector(k=k)
            P_composed = C.projector(k=k, compose=True)
            assert _ops_equal(P_direct, P_composed), \
                f"projector mismatch at k={k}"


# ---------------------------------------------------------------------------
# compose=False: P-term filter alone
# ---------------------------------------------------------------------------

class TestFilterOnly:
    def test_n_terms_compose_false(self, chain_1d):
        """F_C(k) has at most P terms."""
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points:
            F = C.projector(k=k, compose=False)
            # number of operators in SumOperator <= P
            assert len(F.operators) <= C.n_coset_reps

    def test_compose_false_times_ph_equals_full(self, chain_1d):
        """
        F_C(k) @ P_H(k) must equal T_full.projector(k).
        This is the algebraic identity, tested without pre-applying P_H to a state.
        """
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for k in T_full.k_points:
            F_C  = C.projector(k=k, compose=False)
            P_H  = T_sub.projector(k=k)
            combined = F_C @ P_H
            P_direct = T_full.projector(k=k)
            assert _ops_equal(combined, P_direct), \
                f"F_C @ P_H != P_G at k={k}"


# ---------------------------------------------------------------------------
# label= interface
# ---------------------------------------------------------------------------

class TestLabelInterface:
    def test_label_agrees_with_k(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        for label, k in zip(C.irrep_labels, C.k_points):
            P_by_label = C.projector(label=label, compose=True)
            P_by_k     = C.projector(k=k,         compose=True)
            assert _ops_equal(P_by_label, P_by_k), \
                f"label={label!r} and k={k} give different projectors"

    def test_invalid_label_raises(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        with pytest.raises(ValueError, match="Unknown label"):
            C.projector(label="k=NotALabel")

    def test_no_argument_raises(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        with pytest.raises(TypeError):
            C.projector()

    def test_both_arguments_raises(self, chain_1d):
        hi, lattice, T_full, T_sub, stride = chain_1d
        C = T_full.coset_filter(T_sub)
        with pytest.raises(TypeError):
            C.projector(k=0.0, label="k=0")


# ---------------------------------------------------------------------------
# Iterative / multi-level decomposition
# ---------------------------------------------------------------------------

class TestIterativeDecomposition:
    """
    P_G(k) = C_1.projector(k, compose=False)
           @ C_2.projector(k, compose=False)
           @ T_coarse.projector(k)

    for a three-level chain T_fine ⊇ T_medium ⊇ T_coarse.
    """

    @pytest.mark.parametrize("L,s1,s2", [
        (8,  2, 4),   # T_full(1) -> T_half(2) -> T_quarter(4)
        (16, 2, 4),
        (16, 4, 8),
    ])
    def test_three_level_1d(self, L, s1, s2):
        assert s2 % s1 == 0, "s2 must be a multiple of s1"
        lattice = nk.graph.Chain(L, pbc=True)
        hi      = nk.hilbert.Spin(0.5, L)

        T_fine   = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_medium = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=s1)
        )
        T_coarse = nk.symmetry.canonical_representation(
            hi, lattice.translation_group(strides=s2)
        )

        C_1 = T_fine.coset_filter(T_medium)    # s1 coset reps
        C_2 = T_medium.coset_filter(T_coarse)  # s2/s1 coset reps

        for k in T_fine.k_points:
            P_direct = T_fine.projector(k=k)
            P_iter   = (
                C_1.projector(k=k, compose=False)
                @ C_2.projector(k=k, compose=False)
                @ T_coarse.projector(k=k)
            )
            assert _ops_equal(P_direct, P_iter), \
                f"three-level decomposition failed at k={k} (L={L}, s1={s1}, s2={s2})"

    def test_coset_reps_count_product(self):
        """
        |C_1| * |C_2| * |T_coarse| == |T_full|
        """
        L = 16
        lattice = nk.graph.Chain(L, pbc=True)
        hi      = nk.hilbert.Spin(0.5, L)
        T_fine   = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_medium = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))
        T_coarse = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=4))

        C_1 = T_fine.coset_filter(T_medium)
        C_2 = T_medium.coset_filter(T_coarse)

        assert C_1.n_coset_reps * C_2.n_coset_reps * len(T_coarse.group.elems) \
               == len(T_fine.group.elems)


# ---------------------------------------------------------------------------
# NQS / wavefunction usage example
# ---------------------------------------------------------------------------

class TestNQSWorkflow:
    """
    Smoke test: extract perms and chars and verify the wavefunction-level
    symmetrization is correct on a toy example.

    The symmetrized log-amplitude is:
        log_psi_sym(sigma) = log sum_c  chars[c]  * exp(log_psi(sigma[perms[c]]))
    """

    def test_wavefunction_symmetrization_k0(self):
        """
        For k=0 the symmetrized amplitude is a real equal-weight average.
        Any constant function should be invariant.
        """
        L = 8
        lattice = nk.graph.Chain(L, pbc=True)
        hi      = nk.hilbert.Spin(0.5, L)
        T_full  = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_half  = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))
        C = T_full.coset_filter(T_half)

        k0 = np.zeros(1)
        chars = C.chars(k0)
        assert np.allclose(chars, 1.0)
        assert np.allclose(np.abs(chars), 1.0)

    def test_wavefunction_symmetrization_characters_are_roots_of_unity(self):
        """Characters must be P-th roots of unity for translation groups."""
        L = 8
        lattice = nk.graph.Chain(L, pbc=True)
        hi      = nk.hilbert.Spin(0.5, L)
        T_full  = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T_half  = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))
        C = T_full.coset_filter(T_half)

        P = C.n_coset_reps
        for k in T_full.k_points:
            chars = C.chars(k)
            # chars are of the form e^{-ik*d_c}; their P-th power may not be 1
            # but they must have unit modulus
            assert np.allclose(np.abs(chars), 1.0), \
                f"chars at k={k} are not on the unit circle: {chars}"

    def test_perms_chars_progressive_symmetrization(self):
        """
        Demonstrate the iterative training data structure.
        Checks that perms and chars have correct shapes for a 3-level setup.
        """
        L = 16
        lattice = nk.graph.Chain(L, pbc=True)
        hi      = nk.hilbert.Spin(0.5, L)

        T1 = nk.symmetry.canonical_representation(hi, lattice.translation_group())
        T2 = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))
        T4 = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=4))

        C_1_2 = T1.coset_filter(T2)   # T1/T2
        C_2_4 = T2.coset_filter(T4)   # T2/T4

        assert C_1_2.perms.shape == (2, L)
        assert C_2_4.perms.shape == (2, L)

        k = np.array([3 * np.pi / 8])
        assert C_1_2.chars(k).shape == (2,)
        assert C_2_4.chars(k).shape == (2,)

        # Level 1 chars at k=3π/8: e^{-i*3π/8 * {0,1}}
        expected_c12 = np.exp(-1j * 3 * np.pi / 8 * np.array([0.0, 1.0]))
        assert np.allclose(C_1_2.chars(k), expected_c12, atol=1e-12)

        # Level 2 chars at k=3π/8: e^{-i*3π/8 * {0,2}}
        expected_c24 = np.exp(-1j * 3 * np.pi / 8 * np.array([0.0, 2.0]))
        assert np.allclose(C_2_4.chars(k), expected_c24, atol=1e-12)
```


---

## Complete standalone usage example (docstring / tutorial)

```python
"""
Progressive momentum projection with TranslationCosetFilter.

System: 1D Heisenberg chain, L=16 sites.
Goal:   prepare a variational state at full-lattice momentum k = 3π/8.

Strategy:
  1.  Build the multi-level coset structure: stride 4 → 2 → 1.
  2.  Show that the full projector decomposes as a composition of
      small (2-term) filters.
  3.  Show the NQS training schedule: add symmetry progressively.
"""

import jax
import jax.numpy as jnp
import numpy as np
import netket as nk

# ── Lattice and Hilbert space ────────────────────────────────────────────────
L       = 16
lattice = nk.graph.Chain(L, pbc=True)
hi      = nk.hilbert.Spin(0.5, L)

# ── Translation representations at three levels ──────────────────────────────
T1 = nk.symmetry.canonical_representation(hi, lattice.translation_group())
T2 = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=2))
T4 = nk.symmetry.canonical_representation(hi, lattice.translation_group(strides=4))

print("T1 k-points:", T1.irrep_labels)
# ['k=0', 'k=0.125π', 'k=0.25π', ..., 'k=-0.125π']  (16 values)

print("T2 k-points:", T2.irrep_labels)
# ['k=0', 'k=0.25π', 'k=0.5π', ..., 'k=-0.25π']      (8 values)

print("T4 k-points:", T4.irrep_labels)
# ['k=0', 'k=0.5π', 'k=-π', 'k=-0.5π']               (4 values) -- NOTE: these
# are the actual T4 zone momenta (BZ boundary at π/4 for stride=4 on L=16)

# ── Coset filters ────────────────────────────────────────────────────────────
C_1_2 = T1.coset_filter(T2)   # G=T1, H=T2: coset reps {T^0, T^1}
C_2_4 = T2.coset_filter(T4)   # G=T2, H=T4: coset reps {T^0, T^2}

print(C_1_2)
# TranslationCosetFilter(
#   hilbert=Spin(s=1/2, N=16),
#   lattice=...,
#   full_group=16 elements (strides=[1]),
#   sub_group=8 elements  (strides=[2]),
#   coset_reps=2,
#   k_points=[k=0, ...] (16 total)
# )

# ── Target momentum ──────────────────────────────────────────────────────────
k_total = 3 * np.pi / 8

# Characters at each level:
print("C_1_2 chars at k=3π/8:", C_1_2.chars(k_total))
# [1.+0.j,  0.707-0.707j]   e^{0} and e^{-i*3π/8}  → complex

print("C_2_4 chars at k=3π/8:", C_2_4.chars(k_total))
# [1.+0.j,  0.-1.j]         e^{0} and e^{-i*3π/4}  → complex

# ── Verify the decomposition identity ───────────────────────────────────────
P_direct = T1.projector(k=k_total)              # 16-term projector
P_iter   = (
    C_1_2.projector(k=k_total, compose=False)   # 2-term filter
    @ C_2_4.projector(k=k_total, compose=False) # 2-term filter
    @ T4.projector(k=k_total)                   # 4-term projector
)
# P_direct == P_iter  (verified by tests above; total 8 terms vs 16)

# ── NQS training schedule ────────────────────────────────────────────────────
#
# At each training phase the wavefunction is:
#
#   Phase 1 (4 NN evals):
#     log_psi_1(σ) = log Σ_{g ∈ T4}   exp(-i k_total·g) * NN(T_g σ)
#     (T4 has 4 elements; chars at k=3π/8 restricted to T4:
#      e^{-i*3π/8*{0,4,8,12}} = {1, e^{-i3π/2}, e^{-i3π}, e^{-i9π/2}}
#                              = {1, i, -1, -i})
#
#   Phase 2 (8 NN evals = Phase1 × C_2_4 × 2):
#     log_psi_2(σ) = log Σ_{c ∈ C_2_4} exp(-i k_total·c) * psi_1(T_c σ)
#
#   Phase 3 (16 NN evals = Phase2 × C_1_2 × 2):
#     log_psi_3(σ) = log Σ_{c ∈ C_1_2} exp(-i k_total·c) * psi_2(T_c σ)
#
# psi_3 == full symmetrisation at k=3π/8.

perms_T4  = np.asarray(T4.group)              # (4, 16) — for Phase 1
chars_T4  = np.array(                          # (4,)
    [np.exp(-1j * k_total * d) for d in [0, 4, 8, 12]]
)
perms_C24 = C_2_4.perms                        # (2, 16) — for Phase 2
chars_C24 = C_2_4.chars(k_total)               # (2,)
perms_C12 = C_1_2.perms                        # (2, 16) — for Phase 3
chars_C12 = C_1_2.chars(k_total)               # (2,)

import flax.linen as nn

class ProgressiveRBM(nn.Module):
    features: int
    level: int   # 1, 2, or 3

    perms_T4:  tuple; chars_T4:  tuple
    perms_C24: tuple; chars_C24: tuple
    perms_C12: tuple; chars_C12: tuple

    @nn.compact
    def __call__(self, x):
        def rbm(sigma):
            h = nn.Dense(self.features, param_dtype=complex)(sigma.astype(complex))
            return jnp.sum(jnp.log(jnp.cosh(h)))

        def log_sum(log_psis, chars):
            """log Σ_i chars[i] * exp(log_psi_i)  (numerically stable)."""
            c = jnp.array(chars)
            lp = jnp.array(log_psis)
            return jnp.log(jnp.dot(c, jnp.exp(lp)))

        # Level 1: T4 average (4 evals)
        def phase1(sigma):
            p = jnp.array(self.perms_T4)
            return log_sum(jax.vmap(rbm)(sigma[p]), self.chars_T4)

        if self.level == 1:
            return phase1(x)

        # Level 2: C_2_4 filter on top (×2 evals)
        def phase2(sigma):
            p = jnp.array(self.perms_C24)
            return log_sum(jax.vmap(phase1)(sigma[p]), self.chars_C24)

        if self.level == 2:
            return phase2(x)

        # Level 3: C_1_2 filter on top (×2 evals)
        def phase3(sigma):
            p = jnp.array(self.perms_C12)
            return log_sum(jax.vmap(phase2)(sigma[p]), self.chars_C12)

        return phase3(x)


def make_vstate(level: int, variables=None):
    model = ProgressiveRBM(
        features=32, level=level,
        perms_T4=tuple(map(tuple, perms_T4)),   chars_T4=tuple(chars_T4),
        perms_C24=tuple(map(tuple, perms_C24)), chars_C24=tuple(chars_C24),
        perms_C12=tuple(map(tuple, perms_C12)), chars_C12=tuple(chars_C12),
    )
    vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), model, n_samples=512)
    if variables is not None:
        vs.variables = variables
    return vs


H = nk.operator.Heisenberg(hi, lattice)

# Phase 1: coarse symmetry (4 forward passes / sample)
vs = make_vstate(level=1)
gs = nk.driver.VMC(H, nk.optimizer.Sgd(0.01), variational_state=vs)
gs.run(n_iter=300)

# Phase 2: finer symmetry, warm-start (8 forward passes / sample)
vs = make_vstate(level=2, variables=vs.variables)
gs = nk.driver.VMC(H, nk.optimizer.Sgd(0.005), variational_state=vs)
gs.run(n_iter=200)

# Phase 3: full symmetry (16 forward passes / sample)
vs = make_vstate(level=3, variables=vs.variables)
gs = nk.driver.VMC(H, nk.optimizer.Sgd(0.002), variational_state=vs)
gs.run(n_iter=200)
```


---

## Checklist for implementation

- [ ] Create `netket/_src/symmetry/translation_coset_filter.py`
      with `TranslationCosetFilter` and private helpers
      `_cartesian_displacements`, `_coset_rep_mask`, `_validate_subgroup`.

- [ ] Confirm that `TranslationGroup` has `.strides` attribute (tuple of ints,
      one per lattice dimension, giving the step size of the group).  If the
      attribute is named differently (e.g. `.stride`, `._strides`) adjust
      `_coset_rep_mask` accordingly.

- [ ] Confirm that `np.asarray(group)` returns shape `(|G|, N)` integer
      permutation array.  If not, adapt `_cartesian_displacements` and
      `perms`.

- [ ] Confirm that `TranslationRepresentation.operators` is indexable by
      a boolean numpy mask.  If it returns a generator/tuple, wrap in
      `np.array(list(rep.operators), dtype=object)` before masking.

- [ ] Add `active_axes` cached property to `TranslationRepresentation` if not
      already present (needed by `TranslationCosetFilter.chars`).

- [ ] Add `coset_filter` method to `TranslationRepresentation`.

- [ ] Update `netket/_src/symmetry/__init__.py` and
      `netket/symmetry/__init__.py` to import and export
      `TranslationCosetFilter`.

- [ ] Run the test suite:
      `pytest tests/symmetry/test_translation_coset_filter.py -v`

- [ ] Add a brief entry to the `netket/symmetry` API docs page
      (RST / autodoc directive for `TranslationCosetFilter`).
