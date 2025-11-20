import netket as nk
import netket.experimental as nkx
from netket.operator import fermion

import numpy as np
import matplotlib.pyplot as plt

# System parameters
L = 3  # Lattice size (L x L)
D = 2  # Dimension
t = 1.0  # Hopping strength
U = 4.0  # On-site interaction strength

# Create the graph our fermions can hop on
g = nk.graph.Hypercube(length=L, n_dim=D, pbc=True)
n_sites = g.n_nodes

# Create a hilbert space with spin-1/2 fermions
n_fermions_per_spin = (n_sites // 2, n_sites // 2)
hi = nk.hilbert.SpinOrbitalFermions(
    n_sites, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin
)

print(f"System: {L}x{L} lattice with {n_sites} sites")
print(f"Fermions: {n_fermions_per_spin[0]} spin-up, {n_fermions_per_spin[1]} spin-down")
print(f"Parameters: t={t}, U={U}")

# Create the Fermi-Hubbard Hamiltonian
# H = -t sum_{<ij>,sigma} (c_i^dag c_j + h.c.) + U sum_i (n_i_up n_i_down)
spin_values = [-1, 1]

# Hopping term
hop = sum(
    fermion.create(hi, site=i, sz=sz) @ fermion.destroy(hi, site=j, sz=sz)
    + fermion.create(hi, site=j, sz=sz) @ fermion.destroy(hi, site=i, sz=sz)
    for i, j in g.edges()
    for sz in spin_values
)

# On-site interaction term
interaction = sum(
    fermion.number(hi, site=i, sz=spin_values[0])
    @ fermion.number(hi, site=i, sz=spin_values[1])
    for i in range(n_sites)
)

ham = -t * hop + U * interaction
ham = ham.reduce()

# Compute exact ground state energy for comparison
print("\nComputing exact ground state energy...")
ed_energies = np.linalg.eigvalsh(ham.to_dense())
exact_e0 = ed_energies[0]
print(f"Exact ground state energy: {exact_e0:.6f}")

# Initialize the mean-field variational state (Restricted Hartree-Fock)
vstate_mf = nkx.vqs.DeterminantVariationalState(
    hi, generalized=False, restricted=True, seed=42
)

e_init = vstate_mf.expect(ham).mean
print(f"Initial energy: {e_init:.6f}")

# Optimize using gradient descent
print("\nOptimizing with gradient descent...")
opt = nk.optimizer.Sgd(learning_rate=0.05)

energies = [e_init.real]
n_steps = 100
for step in range(n_steps):
    e, grad = vstate_mf.expect_and_grad(ham)
    vstate_mf.parameters = vstate_mf.optimizer_update(opt, grad)
    energies.append(e.mean.real)

    if step % 10 == 0:
        print(f"Step {step:3d}: E = {e.mean.real:.6f}")

print(f"\nFinal mean-field energy: {energies[-1]:.6f}")
error = abs(energies[-1] - exact_e0)
print(f"Error from exact: {error:.6f} ({100 * error / abs(exact_e0):.2f}%)")

# Convert to MCState for optimization with Stochastic Reconfiguration
print("\n" + "=" * 60)
print("Converting to MCState for SR optimization...")
print("=" * 60)

sampler = nk.sampler.MetropolisFermionHop(
    hi, graph=g, n_chains=16, spin_symmetric=True, sweep_size=64
)
vstate_mc = vstate_mf.to_mcstate(sampler, n_samples=512, n_discard_per_chain=10)
print(f"Converted to MCState (n_samples={vstate_mc.n_samples})")

# Optimize with SR using the QGT
print("Optimizing with SR...")
opt_sr = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.1)
gs = nk.driver.VMC(ham, opt_sr, variational_state=vstate_mc, preconditioner=sr)
log = nk.logging.RuntimeLog()
gs.run(50, out=log)

final_e_mc = vstate_mc.expect(ham).mean.real
print(f"Final MC+SR energy: {final_e_mc:.6f}")

# Plot energy convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Mean-field optimization
ax1.plot(energies, "b-", linewidth=2, label="Mean-field (SGD)")
ax1.axhline(exact_e0, color="r", linestyle="--", linewidth=1.5, label="Exact")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Energy")
ax1.set_title("Mean-Field Optimization")
ax1.legend()
ax1.grid(True, alpha=0.3)

# MC+SR optimization
mc_energies = log["Energy"]["Mean"].real
ax2.plot(mc_energies, "g-", linewidth=2, label="MC + SR")
ax2.axhline(exact_e0, color="r", linestyle="--", linewidth=1.5, label="Exact")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Energy")
ax2.set_title("Monte Carlo + SR Optimization")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fermi_hubbard_meanfield.png", dpi=150, bbox_inches="tight")
print(f"\nSaved plot to fermi_hubbard_meanfield.png")
plt.show()
