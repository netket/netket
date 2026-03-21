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
Progressive translation-symmetry refinement via TranslationCosetFilter
and nk.vqs.apply_operator.

System: 1D Heisenberg chain, L=16 sites, target momentum k=0 (Gamma point).

The RBM has no built-in symmetry.  Symmetry is imposed externally by
wrapping the state with a translation-group projector.  We refine the
symmetry progressively, doubling the group at each stage:

  Stage 0  no operator         RBM alone          (no symmetry)
  Stage 1  P_T8               2  translations
  Stage 2  F_{T4/T8} @ P_T8   4  translations   (applies 2-term coset filter on top)
  Stage 3  F_{T2/T4} @ ...     8  translations
  Stage 4  F_{T1/T2} @ ...    16  translations   (full group)

Each stage calls nk.vqs.apply_operator with the next coset filter, which
is automatically composed with the existing operator into a ProductOperator:

    B @ (A @ |psi>)  -->  (B * A) @ |psi>

The base RBM parameters are unchanged between stages and warm-start each
new training phase automatically — no parameter copying needed.
"""

import numpy as np
import netket as nk


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

L = 16
lattice = nk.graph.Chain(L, pbc=True)
hi = nk.hilbert.Spin(0.5, L, total_sz=0)
H = nk.operator.Heisenberg(hilbert=hi, graph=lattice)

# Target momentum k=0 (Gamma point, ground state of the Heisenberg chain).
k_target = np.array([0.0])

# ---------------------------------------------------------------------------
# Translation representations and coset filters
# ---------------------------------------------------------------------------

T1 = nk.symmetry.canonical_representation(hi, lattice.translation_group())  # 16 elems
T2 = nk.symmetry.canonical_representation(
    hi, lattice.translation_group(strides=2)
)  #  8 elems
T4 = nk.symmetry.canonical_representation(
    hi, lattice.translation_group(strides=4)
)  #  4 elems
T8 = nk.symmetry.canonical_representation(
    hi, lattice.translation_group(strides=8)
)  #  2 elems

# One coset filter per refinement step (each has only 2 reps).
# F_{G/H}(k) upgrades an H-symmetric state to a G-symmetric one.
C_4_8 = T4.coset_filter(T8)  # F_{T4/T8}: 2 reps {T^0, T^4}  — T8 → T4
C_2_4 = T2.coset_filter(T4)  # F_{T2/T4}: 2 reps {T^0, T^2}  — T4 → T2
C_1_2 = T1.coset_filter(T2)  # F_{T1/T2}: 2 reps {T^0, T^1}  — T2 → T1
# or equivalently via chaining (each step delegates to sub_rep.coset_filter):
# C_1_2 = T1.coset_filter(T2)
# C_2_4 = T1.coset_filter(T2).coset_filter(T4)
# C_4_8 = T1.coset_filter(T2).coset_filter(T4).coset_filter(T8)
print(
    f"|T8|={len(T8.group.elems)}  |T4|={len(T4.group.elems)}  "
    f"|T2|={len(T2.group.elems)}  |T1|={len(T1.group.elems)}"
)

# ---------------------------------------------------------------------------
# Base variational state (plain RBM, no symmetry)
# ---------------------------------------------------------------------------

model = nk.models.RBM(alpha=2, param_dtype=float)
sampler = nk.sampler.MetropolisExchange(hi, graph=lattice, n_chains=16)
vs = nk.vqs.MCState(sampler, model, n_samples=1024, n_discard_per_chain=16)

print(f"RBM parameters: {vs.n_parameters}\n")

# ---------------------------------------------------------------------------
# Training schedule
#
# At each stage we call nk.vqs.apply_operator with the next operator.
# When the state already has an operator applied, the new operator is
# automatically combined via ProductOperator (no double wrapping).
#
# Effective operator at each stage:
#   Stage 1: P_T8
#   Stage 2: F_{T4/T8} @ P_T8  =  P_T4
#   Stage 3: F_{T2/T4} @ P_T4  =  P_T2
#   Stage 4: F_{T1/T2} @ P_T2  =  P_T1
# ---------------------------------------------------------------------------

schedule = [
    # (operator_or_None,                               label,                        n_iter, lr)
    (None, "Stage 0: no symmetry", 100, 0.02),
    (T8.projector(k=k_target), "Stage 1: T8  (2 translations)", 100, 0.01),
    (
        C_4_8.projector_refinement(k=k_target),
        "Stage 2: T4  (4 translations)",
        100,
        0.01,
    ),
    (
        C_2_4.projector_refinement(k=k_target),
        "Stage 3: T2  (8 translations)",
        100,
        0.005,
    ),
    (
        C_1_2.projector_refinement(k=k_target),
        "Stage 4: T1 (16 translations)",
        100,
        0.005,
    ),
]

vs_cur = vs
for operator, label, n_iter, lr in schedule:
    print(f"--- {label} ---")

    if operator is not None:
        # Wrap the current state with the new operator.
        # If vs_cur already has an operator, the two are combined automatically.
        vs_cur = nk.vqs.apply_operator(operator, vs_cur)

    optimizer = nk.optimizer.Sgd(learning_rate=lr)
    gs = nk.driver.VMC(H, optimizer, variational_state=vs_cur)
    gs.run(n_iter=n_iter, out=None)

    E = vs_cur.expect(H)
    print(f"    E/site = {E.mean.real / L:.6f} ± {E.error_of_mean.real / L:.6f}\n")

print("Done.")
