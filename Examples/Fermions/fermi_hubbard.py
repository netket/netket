import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
from netket.operator.fermion import create, destroy, number

L = 2  # take a 2x2 lattice
D = 2
t = 1  # tunneling/hopping
U = 0.01  # coulomb

# create the graph our fermions can hop on
g = nk.graph.Hypercube(length=L, n_dim=D, pbc=True)
Nsites = g.n_nodes

# create a hilbert space with 2 up and 2 down spins
hi = nk.hilbert.SpinOrbitalFermions(Nsites, s=1 / 2, n_fermions=(2, 2))

# create an operator representing fermi hubbard interactions
# -t (i^ j + h.c.) + U (i^ i j^ j)
c = lambda site, sz: create(hi, site, sz=sz)
cdag = lambda site, sz: destroy(hi, site, sz=sz)
nc = lambda site, sz: number(hi, site, sz=sz)
up = +1 / 2
down = -1 / 2
ham = []
for u, v in g.edges():
    for sz in (up, down):
        ham.append(-t * cdag(u, sz) * c(v, sz) - t * cdag(v, sz) * c(u, sz))
        ham.append(U * nc(u, sz) * nc(v, sz))
ham = sum(ham)

# create everything necessary for the VMC

# move the fermions around
sa = nk.sampler.MetropolisExchange(hi, graph=g)

# since the hilbert basis is a set of occupation numbers, we can take a general NN
ma = nk.models.RBM(alpha=1, dtype=complex)
vs = nk.vqs.MCState(sa, ma, n_discard_per_chain=100, n_samples=512)

# we will use sgd with Stochastic R
opt = nk.optimizer.Sgd(learning_rate=0.001)
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
