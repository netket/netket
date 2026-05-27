# Copyright 2021 The NetKet Authors - All rights reserved.
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

import numpy as np
import matplotlib.pyplot as plt
import netket as nk
from netket import experimental as nkx
import optax

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer with a decreasing learning rate
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

# Variational monte carlo driver with a variational state
gs = nk.driver.VMC_SR(
    ha,
    op,
    variational_state=vs,
    diag_shift=0.01,
)

# Run the optimization for 500 iterations
gs.run(n_iter=500, out="test", timeit=True)

# Staggered connected σz-σz correlators C(r) = (-1)^r (<σz_0 σz_r> - <σz_0><σz_r>)
sz = [nk.operator.spin.sigmaz(hi, i) for i in range(L)]
correlator_ops = [
    nk.observable.ConnectedCorrelator(sz[0], (-1) ** r * sz[r]) for r in range(L)
]

## Compute observables in different symmetry subsectors

# --- unmodified state ---
print("Computing correlators [unprojected]...")
stats_raw = vs.expect_to_precision(correlator_ops, atol=5e-2)
C_raw, C_raw_err = zip(*((s.mean.real, s.error_of_mean.real) for s in stats_raw))

# --- project onto k=0 translation sector (without retraining) ---
trans_rep = nk.symmetry.canonical_representation(hi, g.translation_group())
vs_k0 = trans_rep.project(vs, k=0)
print("Computing correlators [k=0 projected]...")
stats_k0 = vs_k0.expect_to_precision(correlator_ops, atol=5e-2)
C_k0, C_k0_err = zip(*((s.mean.real, s.error_of_mean.real) for s in stats_k0))

# --- project onto k=0 ∩ even/odd parity sectors (without retraining) ---
sf_rep = nk.symmetry.spin_flip_representation(hi)
vs_even = sf_rep.project(vs_k0, label="+1")
print("Computing correlators [k=0, even parity]...")
stats_even = vs_even.expect_to_precision(correlator_ops, atol=5e-2)
C_even, C_even_err = zip(*((s.mean.real, s.error_of_mean.real) for s in stats_even))

vs_odd = sf_rep.project(vs_k0, label="-1")
print("Computing correlators [k=0, odd parity]...")
stats_odd = vs_odd.expect_to_precision(correlator_ops, atol=5e-2)
C_odd, C_odd_err = zip(*((s.mean.real, s.error_of_mean.real) for s in stats_odd))

# --- energies to atol=1e-3 ---
print("\nEnergies:")
for label, vstate in [
    ("unprojected", vs),
    ("k=0 projected", vs_k0),
    ("k=0, even parity", vs_even),
    ("k=0, odd parity", vs_odd),
]:
    e = vstate.expect_to_precision(ha, atol=1e-3, verbose=False)
    print(f"  [{label}]: {e}")

# --- plot ---
r = np.arange(L)
fig, ax = plt.subplots()

curves = [
    (C_raw, C_raw_err, "unprojected", "C0"),
    (C_k0, C_k0_err, "k=0 projected", "C1"),
    (C_even, C_even_err, "k=0, even parity", "C2"),
    (C_odd, C_odd_err, "k=0, odd parity", "C3"),
]

for C, err, label, color in curves:
    ax.errorbar(r, C, yerr=err, fmt="o-", capsize=3, label=label, color=color)

ax.set_xlabel("r")
ax.set_ylabel(r"$(-1)^r\langle \sigma^z_0 \sigma^z_r \rangle_c$")
ax.set_title("Connected σz-σz correlator (Ising 1D, h=1.0)")
ax.axhline(0, color="k", linewidth=0.5)
ax.legend()
plt.tight_layout()
plt.savefig("correlators.pdf")
plt.show()
