# Implementation Plan: `LabeledRepresentationCosetFilter`

## Overview

Extend the coset-filter machinery from translation groups to **arbitrary finite
groups** represented as `LabeledRepresentation` (and its subclasses), enabling
point-group refinement via the same progressive `apply_operator` workflow.

The new class `LabeledRepresentationCosetFilter` generalises
`TranslationCosetFilter`:

| Feature                  | `TranslationCosetFilter`       | `LabeledRepresentationCosetFilter` |
|--------------------------|--------------------------------|------------------------------------|
| Group type               | `TranslationGroup`             | any `PermutationGroup`             |
| Coset-rep selection      | stride-based (fast)            | generic group-theory algorithm     |
| Characters χ_ρ(c)        | `exp(-ik·d_c)`                 | from character table (row `ρ`)     |
| Irrep dimension          | always 1                       | may be > 1 (E irrep of D4, etc.)  |
| Factory                  | `TranslationRepresentation.coset_filter` | `LabeledRepresentation.coset_filter` |
| Chaining `.coset_filter` | ✓ (already implemented)        | needs adding                       |


---

## Mathematical Background

### General coset filter

Let G be a finite group, H ≤ G a subgroup, and ρ an irrep of G with dimension
d_ρ and character χ_ρ.  The G-projector onto sector ρ is

    P_G(ρ) = (d_ρ / |G|)  Σ_{g ∈ G}  χ_ρ(g)*  R_g

where R_g is the operator for group element g.

The coset decomposition G = C × H (disjoint union of left-cosets) gives

    P_G(ρ) = F_C(ρ) @ P_H(ρ|_H)

where

    F_C(ρ) = (d_ρ / |C|)  Σ_{c ∈ C}  χ_ρ(c)*  R_c     (|C| = |G|/|H| terms)
    P_H(ρ|_H) = Σ_μ  (d_μ/|H|)  Σ_{h ∈ H}  χ_μ(h)*  R_h   (sub-group projector)

For a 1D irrep (d_ρ = 1, all point-group irreps of D4 except E) this is
identical in structure to the translation case.  For the 2D irrep E of D4,
d_ρ = 2 and the formula still holds.

### Iterative decomposition (chain G ⊇ H₁ ⊇ H₂ ⊇ … ⊇ Hₙ)

    P_G(ρ) = F_{G/H₁}(ρ) @ F_{H₁/H₂}(ρ) @ … @ P_{Hₙ}(ρ|_{Hₙ})

Each filter F_{Hᵢ₋₁/Hᵢ}(ρ) uses only |Hᵢ₋₁|/|Hᵢ| terms.


---

## New Public API

### `LabeledRepresentationCosetFilter`

```python
class LabeledRepresentationCosetFilter:
    """
    Coset Fourier filter for G/H, where G and H are finite groups represented
    as LabeledRepresentation objects.

    Generalises TranslationCosetFilter to arbitrary finite groups (point groups,
    space groups, etc.).

    Usage::

        C = full_rep.coset_filter(sub_rep)
        F = C.projector(label="(+1, +1)", compose=False)  # |C| terms
        P = C.projector(label="(+1, +1)", compose=True)   # |C| + |H| terms
        # C.projector(compose=True)  ==  full_rep.projector(label=...)

    Chaining::

        C_c2c1 = rep_d4.coset_filter(rep_c4)
        C_c4c2 = rep_d4.coset_filter(rep_c4).coset_filter(rep_c2)
        C_c2c1_again = rep_d4.coset_filter(rep_c4).coset_filter(rep_c2).coset_filter(rep_c1)
    """
```

#### Constructor

```python
def __init__(
    self,
    full_rep: LabeledRepresentation,
    sub_rep:  LabeledRepresentation,
) -> None
```

Validates:
1. Same Hilbert space.
2. `sub_rep.group` is a subgroup of `full_rep.group` (every element of
   `sub_rep.group` appears in `full_rep.group`).
3. `|full_rep.group|` is divisible by `|sub_rep.group|`.
4. Not identical groups.

#### Key properties

```python
@cached_property
def n_coset_reps: int     # |G| / |H|

@cached_property
def perms: np.ndarray     # (P, N) int  — permutation arrays for coset reps

@cached_property
def irrep_labels: list[str]   # from full_rep

@cached_property
def irrep_dims: np.ndarray    # (n_irreps,) int — d_ρ for each irrep
```

#### `chars(label_or_index) -> np.ndarray`

Returns `χ_ρ(c)` for each coset rep c.  Shape `(P,)` for 1D irreps,
`(P,)` using the character (trace of matrix rep) for multi-dim irreps.

#### `projector(label=, compose=, atol=) -> SumOperator`

Same interface as `TranslationCosetFilter.projector`.

#### `coset_filter(subgroup) -> LabeledRepresentationCosetFilter`

Chains to the next level: returns `self.sub_rep.coset_filter(subgroup)`.

---

## Factory method on `LabeledRepresentation`

```python
def coset_filter(
    self,
    subgroup: "LabeledRepresentation",
) -> "LabeledRepresentationCosetFilter":
    """
    Build the coset Fourier filter for this group G modulo subgroup H.

    Works for any LabeledRepresentation, including point groups and
    (for backward compatibility) TranslationRepresentation.

    Example — D4 point group chain::

        rep_d4 = nk.symmetry.canonical_representation(hi, lattice.point_group())
        rep_c4 = nk.symmetry.canonical_representation(hi, c4_subgroup)
        rep_c2 = nk.symmetry.canonical_representation(hi, c2_subgroup)

        C_d4_c4 = rep_d4.coset_filter(rep_c4)         # F_{D4/C4}: 2 reps
        C_c4_c2 = rep_d4.coset_filter(rep_c4).coset_filter(rep_c2)  # F_{C4/C2}: 2 reps
    """
    from netket._src.symmetry.labeled_representation_coset_filter import (
        LabeledRepresentationCosetFilter,
    )
    return LabeledRepresentationCosetFilter(self, subgroup)
```

Note: `TranslationRepresentation.coset_filter` should remain for backward
compatibility and continue to return a `TranslationCosetFilter` (which is
faster due to stride-based coset rep selection).


---

## Implementation Details

### File layout

```
netket/_src/symmetry/
    labeled_representation_coset_filter.py   ← NEW
    labeled_representation.py                ← add .coset_filter() method
    translation_coset_filter.py              ← add .coset_filter() to TranslationCosetFilter

netket/symmetry/__init__.py                  ← export LabeledRepresentationCosetFilter
```

### Core algorithm: generic coset representative selection

The stride trick used in `TranslationCosetFilter` is specific to cyclic
groups.  For a general `PermutationGroup` G and subgroup H, the coset reps
can be found by a greedy sweep:

```python
def _generic_coset_rep_mask(full_group, sub_group) -> np.ndarray:
    """
    Boolean mask over full_group.elems selecting one representative
    per left-coset of full_group / sub_group.

    A left-coset of g is { g * h : h ∈ H }.
    In terms of inverse-permutation arrays (as returned by to_array()):
        (g * h)_inv = g_inv[h_inv]   (numpy indexing)
    """
    G_arr = np.asarray(full_group)    # (|G|, N)  inverse perm arrays
    H_arr = np.asarray(sub_group)     # (|H|, N)

    # Set of all H elements as frozensets for O(1) membership test
    covered = set()
    mask = np.zeros(len(G_arr), dtype=bool)

    for i, g_inv in enumerate(G_arr):
        key = tuple(g_inv)
        if key in covered:
            continue
        # g is a new coset representative
        mask[i] = True
        # Mark the entire coset { g * h : h ∈ H } as covered
        for h_inv in H_arr:
            covered.add(tuple(g_inv[h_inv]))

    return mask
```

This is O(|G| · |H| · N) in time and O(|G| · N) in space, acceptable for
the small point groups encountered in practice (|D4| = 8, |O_h| = 48).

For large groups (space groups with hundreds of elements) a hash-based
acceleration using numpy structured arrays may be worthwhile.

### Subgroup validation

```python
def _validate_labeled_subgroup(full_rep, sub_rep):
    # 1. Same Hilbert space
    if full_rep.hilbert != sub_rep.hilbert: ...

    # 2. sub_group ≤ full_group: every element of sub appears in full
    G_set = set(map(tuple, np.asarray(full_rep.group)))
    H_arr = np.asarray(sub_rep.group)
    for h in H_arr:
        if tuple(h) not in G_set:
            raise ValueError("sub_rep is not a subgroup of full_rep ...")

    # 3. Lagrange: |G| divisible by |H|
    if len(full_rep.group.elems) % len(sub_rep.group.elems) != 0:
        raise ValueError("...")

    # 4. Not identical
    if len(full_rep.group.elems) == len(sub_rep.group.elems):
        raise ValueError("sub_rep is identical to full_rep ...")
```

### Character computation

For a `LabeledRepresentation`, `group.character_table()` returns an array
of shape `(n_irreps, |G|)` — BUT the columns correspond to conjugacy-class
representatives, not to the ordered elements of `group.elems`.

We need χ_ρ(c) for each coset rep c:

```python
def _chars_for_coset_reps(full_rep, coset_rep_mask, irrep_label):
    """Returns χ_ρ(c) for each selected coset rep, as complex array (P,)."""
    idx = full_rep.irrep_labels.index(irrep_label)

    # character_table()[idx] gives χ_ρ for each GROUP ELEMENT (not just class reps)
    # Check: does LabeledRepresentation expose per-element characters?
    # If not, we need: chars_per_element[i] = CT[idx, class_of_elem_i]
    ct, class_labels, class_idx = full_rep.group.character_table_by_class()
    # class_idx[i] = conjugacy class index of group element i (ordered as elems)
    chars_all = ct[idx, class_idx]            # (|G|,) complex
    return chars_all[coset_rep_mask]          # (P,)  complex
```

**Key open question**: `PermutationGroup.character_table()` returns a
`(n_irreps, n_classes)` array indexed by class, not by element.
`character_table_by_class()` (if it exists) or building the per-element
table by composing with `conjugacy_classes` is needed.  Check the existing
`LabeledRepresentation.projector` to see how it already handles this, and
reuse that logic.

### Multi-dimensional irreps (E irrep of D4, etc.)

For d_ρ > 1, the projector formula gains a prefactor:

    P_G(ρ) = (d_ρ / |G|) Σ_g χ_ρ(g)* R_g
    F_C(ρ) = (d_ρ / |C|) Σ_{c ∈ C} χ_ρ(c)* R_c

The `projector` method should read `irrep_dim` from the character table
(it is the character of the identity element: `χ_ρ(Id) = d_ρ`).

```python
@cached_property
def _irrep_dims(self) -> np.ndarray:
    """d_ρ = χ_ρ(Id) for each irrep."""
    ct = self._full_rep.group.character_table()   # (n_irreps, n_classes)
    # Identity is always the first element in conjugacy class 0
    return ct[:, 0].real.astype(int)
```

The compose=True check (does F_C @ P_H == P_G?) still holds for multi-dim
irreps because the math is the same.

### `projector` method

```python
def projector(self, *, label, compose=True, atol=1e-15):
    idx  = self.irrep_labels.index(label)
    d_rho = self._irrep_dims[idx]

    chars = self._chars_for_irrep(idx)    # (P,) complex
    coeffs = d_rho * np.conj(chars) / self.n_coset_reps
    ops = np.array(list(self._full_rep.operators), dtype=object)[self._coset_mask]
    mask = ~np.isclose(coeffs, 0.0, atol=atol)
    F_C = SumOperator(*ops[mask], coefficients=coeffs[mask])

    if not compose:
        return F_C
    P_H = self._sub_rep.projector(label=self._sub_label(label))
    return F_C @ P_H
```

The `_sub_label(label)` helper maps a full-group irrep label to the
corresponding sub-group irrep label (the restriction ρ|_H).  For the A1
irrep of D4 restricted to C4, the restriction is still A (trivial), and
so on.  For the E irrep this becomes reducible — the restriction should be
handled carefully.

**Simple first implementation**: require the user to supply `sub_label`
explicitly if `compose=True` for a non-trivial restriction, or default to
the trivial irrep of H when `compose=True`.


---

## Tests

File: `tests/symmetry/test_labeled_representation_coset_filter.py`

Key tests:
1. `C.projector(label, compose=True) == full_rep.projector(label)` for all
   irreps of D4, C4, C2 subgroup chains.
2. `n_coset_reps` correct for each pair.
3. `chars(label)` has correct values (compare with character table entries).
4. Multi-level decomposition: `F_{D4/C4} @ F_{C4/C2} @ P_{C2}` == `P_{D4}`.
5. Chaining `.coset_filter` gives the right objects.
6. Validation errors (non-subgroup, different Hilbert, identical groups).


---

## Open questions / design decisions

1. **`character_table_by_class` API**: Does `PermutationGroup` already expose
   per-element characters, or do we need to build them from
   `conjugacy_classes` + `character_table`?  Inspect
   `LabeledRepresentation.projector` source for the existing pattern.

2. **Restriction ρ|_H for compose=True**: For point groups with multi-dim
   irreps, the restriction ρ|_H may be reducible.  The compose=True path
   needs to handle this (at minimum by raising a clear error and documenting
   that compose=False is always safe).

3. **Interleaving translation and point-group stages**: The full space-group
   projector decomposes as `P_{SG} = P_{PG} @ P_{T}` when the little group
   has no mixing.  Exposing a `SpaceGroupCosetFilter` that wraps both is a
   possible future extension; not required for the initial implementation.

4. **`TranslationCosetFilter` vs `LabeledRepresentationCosetFilter`**: Should
   `TranslationRepresentation.coset_filter` return a
   `LabeledRepresentationCosetFilter` (unified class) or keep returning a
   `TranslationCosetFilter` (faster for translations)?  Recommend keeping
   `TranslationCosetFilter` as a specialised fast path and sharing only the
   interface (duck typing, no inheritance required).


---

## Checklist

- [ ] Add `_generic_coset_rep_mask` to a new helper module or to
      `labeled_representation_coset_filter.py`.
- [ ] Implement `_validate_labeled_subgroup`.
- [ ] Implement `LabeledRepresentationCosetFilter` with `perms`, `chars`,
      `projector`, `coset_filter` (chain).
- [ ] Add `coset_filter` factory method to `LabeledRepresentation`.
- [ ] Add `coset_filter` chain method to `LabeledRepresentationCosetFilter`.
- [ ] Update `netket/symmetry/__init__.py`.
- [ ] Write tests covering D4 ⊇ C4 ⊇ C2 subgroup chain.
- [ ] Uncomment and complete the point-group stages in
      `Examples/TranslationCosetAnnealing/coset_annealing_2d.py`.
