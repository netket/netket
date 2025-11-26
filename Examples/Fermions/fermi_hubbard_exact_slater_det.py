import netket as nk
import netket.experimental as nkx
from netket.operator import fermion

import numpy as np
import matplotlib.pyplot as plt

# System parameters
Lx = 16  # Lattice width
Ly = 16  # Lattice height
t = 1.0  # Hopping strength
U = 3.0  # On-site interaction strength

g = nk.graph.Grid(extent=[Lx, Ly], pbc=True)
n_sites = g.n_nodes

# Create a hilbert space with spin-1/2 fermions
# 1/8 doping: 56 total fermions (28 spin-up, 28 spin-down)
n_fermions_per_spin = (28, 28)
hi = nk.hilbert.SpinOrbitalFermions(
    n_sites, s=1 / 2, n_fermions_per_spin=n_fermions_per_spin
)

print(f"System: {Lx}x{Ly} lattice with {n_sites} sites")
print(f"Fermions: {n_fermions_per_spin[0]} spin-up, {n_fermions_per_spin[1]} spin-down")
print(f"Parameters: t={t}, U={U}")
print()

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

# Initialize the mean-field variational state (Generalized Hartree-Fock)
vstate_mf = nkx.vqs.DeterminantVariationalState(hi, generalized=True, seed=42)

e_init = vstate_mf.expect(ham).mean
print(f"Initial energy: {e_init:.6f}")

print("\nOptimizing with gradient descent...")
opt = nk.optimizer.Sgd(learning_rate=0.1)
vmc = nk.driver.VMC(ham, optimizer=opt, variational_state=vstate_mf)
log = nk.logging.RuntimeLog()
vmc.run(2000, out=log)

# Could convert to MCState:
# sampler = nk.sampler.MetropolisFermionHop(
#     hi, graph=g, n_chains=16, spin_symmetric=True, sweep_size=64
# )
# vstate_mc = vstate_mf.to_mcstate(sampler, n_samples=512, n_discard_per_chain=10)

energies = log.data["Energy"]["Mean"]
print(f"Final mean-field energy: {energies[-1]:.6f}")

# Compute charge and spin densities
print("\nComputing charge and spin densities...")
charge_density = np.zeros(n_sites)
spin_density = np.zeros(n_sites)

for i in range(n_sites):
    n_up = fermion.number(hi, site=i, sz=spin_values[0])
    n_down = fermion.number(hi, site=i, sz=spin_values[1])

    charge_density[i] = vstate_mf.expect((n_up + n_down).to_normal_order()).mean.real
    spin_density[i] = vstate_mf.expect((n_up - n_down).to_normal_order()).mean.real

charge_density = charge_density.reshape(Lx, Ly)
spin_density = spin_density.reshape(Lx, Ly)

# Plot results
fig = plt.figure(figsize=(15, 4))

# Energy convergence
ax1 = plt.subplot(1, 3, 1)
ax1.plot(energies, "b-", linewidth=2)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Energy")
ax1.set_title("Mean-Field Optimization")
ax1.grid(True, alpha=0.3)

# Charge density
ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(charge_density.T, cmap="RdBu", origin="lower", aspect="equal")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("Charge Density")
plt.colorbar(im2, ax=ax2, label=r"$\langle n_\uparrow + n_\downarrow \rangle$")

# Spin density
ax3 = plt.subplot(1, 3, 3)
im3 = ax3.imshow(spin_density.T, cmap="RdBu", origin="lower", aspect="equal")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("Spin Density")
plt.colorbar(im3, ax=ax3, label=r"$\langle n_\uparrow - n_\downarrow \rangle$")

plt.tight_layout()
plt.savefig("fermi_hubbard_meanfield_large.png", dpi=150, bbox_inches="tight")
print(f"\nSaved plot to fermi_hubbard_meanfield_large.png")
plt.show()
