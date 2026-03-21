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

"""
Progressive symmetry refinement on a 4×4 Heisenberg lattice — M point k=(π,π).

Demonstrates three things:

1. COST: full projectors at each translation level have |G| terms.  The coset
   approach adds only 2 terms per stage, halving the group each time.

2. EQUIVALENCE:  P_T11(k)  ==  F_x @ F_y @ P_T22(k)
   Verified on a random state by comparing <H> expectation values.

3. TRAINING: 7 stages from bare RBM to full D4 ⋉ T symmetry at k=(π,π).

Translation coset chain (k = (π,π)):
    F_x @ F_y @ P_T22  =  P_T11(k)   (16 translations total)

Point-group coset chain (M point / A1 sector):
    F_{D4/C4} @ F_{C4/C2} @ P_C2(A)  =  P_D4(A1)
"""

import numpy as np
import netket as nk
from netket.utils.group import PermutationGroup


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

L = 4
lattice = nk.graph.Square(L, pbc=True)
hi = nk.hilbert.Spin(0.5, L * L, total_sz=0)
H = nk.operator.Heisenberg(hilbert=hi, graph=lattice)

k_target = np.array([np.pi, np.pi])  # M point (antiferromagnetic wavevector)


# ---------------------------------------------------------------------------
# Translation representations at three levels
#   T_22 ≤ T_21 ≤ T_11  (4 ≤ 8 ≤ 16 elements)
# ---------------------------------------------------------------------------

T_11 = nk.symmetry.canonical_representation(hi, lattice.translation_group())
T_21 = nk.symmetry.canonical_representation(
    hi, lattice.translation_group(strides=(2, 1))
)
T_22 = nk.symmetry.canonical_representation(
    hi, lattice.translation_group(strides=(2, 2))
)

# ---------------------------------------------------------------------------
# 1. Full projectors — term count grows with group size (wasteful at scale)
# ---------------------------------------------------------------------------

print("Full projector term counts at k=(π,π):")
for name, rep in [("T_22", T_22), ("T_21", T_21), ("T_11", T_11)]:
    n = len(rep.projector(k=k_target).operators)
    print(f"  P_{name}: {n:2d} terms")
print()

# ---------------------------------------------------------------------------
# Coset filters — 2 terms each regardless of group size
# ---------------------------------------------------------------------------
#   C_y = F_{T21/T22}: adds y unit-cell step
#   C_x = F_{T11/T21}: adds x unit-cell step

C_y = T_11.coset_filter(T_21).coset_filter(T_22)  # delegates to T_21.coset_filter(T_22)
C_x = T_11.coset_filter(T_21)

P_T22_seed = T_22.projector(k=k_target)
F_y = C_y.projector_refinement(k=k_target)
F_x = C_x.projector_refinement(k=k_target)

print("Coset refinement term counts at k=(π,π):")
print(f"  P_T22  (seed) : {len(P_T22_seed.operators)} terms")
print(f"  F_y (T21/T22) : {len(F_y.operators)} terms")
print(f"  F_x (T11/T21) : {len(F_x.operators)} terms")
print(
    f"  Total staged  : {len(P_T22_seed.operators) + len(F_y.operators) + len(F_x.operators)}"
    f"  vs {len(T_11.projector(k=k_target).operators)} for P_T11 alone"
)
print()

# ---------------------------------------------------------------------------
# 2. Equivalence check:  P_T11  ==  F_x @ F_y @ P_T22  (same RBM params)
# ---------------------------------------------------------------------------

model = nk.models.RBM(alpha=2, param_dtype=float)
sampler = nk.sampler.MetropolisExchange(hi, graph=lattice, n_chains=16)
vs_base = nk.vqs.MCState(sampler, model, n_samples=2048, n_discard_per_chain=32)

vs_direct = nk.vqs.apply_operator(T_11.projector(k=k_target), vs_base)
vs_staged = nk.vqs.apply_operator(
    C_x.projector_refinement(k=k_target),
    nk.vqs.apply_operator(
        C_y.projector_refinement(k=k_target),
        nk.vqs.apply_operator(T_22.projector(k=k_target), vs_base),
    ),
)

E_direct = vs_direct.expect(H)
E_staged = vs_staged.expect(H)

print("Equivalence: P_T11  vs  F_x @ F_y @ P_T22  (same RBM params, should match)")
print(
    f"  Direct  P_T11 : E/site = {E_direct.mean.real / (L*L):.5f}"
    f" ± {E_direct.error_of_mean.real / (L*L):.5f}"
)
print(
    f"  Staged Fx·Fy·P : E/site = {E_staged.mean.real / (L*L):.5f}"
    f" ± {E_staged.error_of_mean.real / (L*L):.5f}"
)
print()

# ---------------------------------------------------------------------------
# Point-group representations  (C2 ≤ C4 ≤ D4)
# ---------------------------------------------------------------------------

_pg = lattice.point_group()
_elems = list(_pg.elems)

_c2_elems = [e for e in _elems if str(e) in ("Id()", "Rot(180°)")]
_c4_elems = [
    e for e in _elems if str(e) in ("Id()", "Rot(90°)", "Rot(180°)", "Rot(-90°)")
]

rep_c2 = nk.symmetry.canonical_representation(
    hi, PermutationGroup(_c2_elems, degree=L * L), warn=False
)
rep_c4 = nk.symmetry.canonical_representation(
    hi, PermutationGroup(_c4_elems, degree=L * L), warn=False
)
rep_d4 = nk.symmetry.canonical_representation(hi, _pg, warn=False)

# irrep_labels[0] is always the trivial (all-ones) irrep
label_c2 = rep_c2.irrep_labels[0]
label_c4 = rep_c4.irrep_labels[0]
label_d4 = rep_d4.irrep_labels[0]

C_c4c2 = rep_d4.coset_filter(rep_c4).coset_filter(rep_c2)  # F_{C4/C2}
C_d4c4 = rep_d4.coset_filter(rep_c4)  # F_{D4/C4}

print(f"Point-group trivial irreps: C2={label_c2!r}, C4={label_c4!r}, D4={label_d4!r}")
print()


# ---------------------------------------------------------------------------
# 3. Training: 7 stages, M point (π,π)
# ---------------------------------------------------------------------------


def run_stage(vs_cur, operator, label, n_iter=100, lr=0.01):
    if operator is not None:
        vs_cur = nk.vqs.apply_operator(operator, vs_cur)
    gs = nk.driver.VMC(H, nk.optimizer.Sgd(learning_rate=lr), variational_state=vs_cur)
    gs.run(n_iter=n_iter, out=None)
    E = vs_cur.expect(H)
    print(
        f"  {label:58s}  E/site = {E.mean.real / (L*L):.5f}"
        f" ± {E.error_of_mean.real / (L*L):.5f}"
    )
    return vs_cur


print("--- Training schedule: M point k=(π,π) ---")
vs_cur = vs_base
vs_cur = run_stage(vs_cur, None, "Stage 0: no symmetry", n_iter=100, lr=0.02)
vs_cur = run_stage(
    vs_cur,
    T_22.projector(k=k_target),
    "Stage 1: T(2,2)  —  4 translations",
    n_iter=100,
    lr=0.01,
)
vs_cur = run_stage(
    vs_cur,
    C_y.projector_refinement(k=k_target),
    "Stage 2: +y unit step  (8 total)",
    n_iter=100,
    lr=0.01,
)
vs_cur = run_stage(
    vs_cur,
    C_x.projector_refinement(k=k_target),
    "Stage 3: +x unit step (16 total)",
    n_iter=100,
    lr=0.005,
)
vs_cur = run_stage(
    vs_cur,
    rep_c2.projector(label=label_c2),
    "Stage 4: +C2  (Rot 180°)",
    n_iter=100,
    lr=0.005,
)
vs_cur = run_stage(
    vs_cur,
    C_c4c2.projector_refinement(label=label_c4),
    "Stage 5: +C4  (Rot ±90°)",
    n_iter=100,
    lr=0.002,
)
vs_cur = run_stage(
    vs_cur,
    C_d4c4.projector_refinement(label=label_d4),
    "Stage 6: +D4  (reflections, full group)",
    n_iter=100,
    lr=0.002,
)

print()
print("Done.")
