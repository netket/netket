import netket as nk
import openfermion
import numpy as np
import matplotlib.pyplot as plt
import json

L = 2  # take a 2x2 lattice
t = 1  # tunneling/hopping
V = 0.01  # coulomb
Nsites = L * L

# use openfermion to create the spinless hubbard model
ham_of = openfermion.hamiltonians.fermi_hubbard(
    L, L, t, V, periodic=True, spinless=False
)

# for now, we will use the same technique to construct a hamiltonian sampler of all possible hopping terms
ham_hopping_of = openfermion.hamiltonians.fermi_hubbard(
    L, L, 1, 0, periodic=True, spinless=False
)

# create a hilbert space with 2 up and 2 down spins
hi = nk.hilbert.SpinOrbitalFermions(Nsites, n_fermions_per_spin=(2, 2))

# create everything necessary for the VMC
ham = nk.operator.FermionOperator2nd.from_openfermion(hi, ham_of)

# for now, we do not have a custom sampler, but we can take the hopping terms
ham_hopping = nk.operator.FermionOperator2nd.from_openfermion(hi, ham_hopping_of)
sa = nk.sampler.MetropolisHamiltonian(hi, ham_hopping)

# since the hilbert basis is a set of occupation numbers, we can take a general NN
ma = nk.models.RBM(alpha=1, dtype=complex)
vs = nk.vqs.MCState(sa, ma, n_discard_per_chain=100, n_samples=512)

# we will use sgd with Stochastic R
opt = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

gs = nk.driver.VMC(ham, opt, variational_state=vs, preconditioner=sr)

# now run the optimization
# first step will take longer in order to compile
exp_name = "fermions_test"
gs.run(500, out=exp_name)

############## plot #################

ed_energies = np.linalg.eigvalsh(ham.to_dense())

with open("{}.log".format(exp_name), "r") as f:
    data = json.load(f)

x = data["Energy"]["iters"]
y = data["Energy"]["Mean"]["real"]

# plot the energy levels
plt.axhline(ed_energies[0], color="red", label="E0")
for e in ed_energies[1:]:
    plt.axhline(e, color="black")
plt.plot(x, y, color="red", label="VMC")
plt.xlabel("step")
plt.ylabel("E")
plt.show()
