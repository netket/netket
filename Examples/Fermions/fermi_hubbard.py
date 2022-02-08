import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json

L = 2  # take a 2x2 lattice
D = 2
t = 1  # tunneling/hopping
U = 0.01  # coulomb

# create the graph our fermions can hop on
g = nk.graph.Hypercube(length=L, n_dim=D, pbc=True)
Nsites = g.n_nodes

# create a hilbert space with 2 up and 2 down spins
hi = nk.hilbert.SpinOrbitalFermions(Nsites, n_fermions_per_spin=(2, 2))

# create an operator representing fermi hubbard interactions
# -t (i^ j + h.c.) + V (i^ i j^ j)
c = lambda site: nk.operator.FermionOperator2nd.create(hi, site)
cdag = lambda site: nk.operator.FermionOperator2nd.destroy(hi, site)
nc = lambda site: nk.operator.FermionOperator2nd.number(hi, site)
ham = []
for u, v in g.edges():
    hopping = -t * cdag(u) * c(v) - t * cdag(v) * c(u)
    coulomb = U * nc(u) * nc(v)
    ham.append(hopping + coulomb)
ham = sum(ham)

# create everything necessary for the VMC

# move the fermions around
sa = nk.sampler.MetropolisExchange(hi, graph=g)

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
