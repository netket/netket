import netket as nk
import numpy as np
import json
from math import pi

L = 10
# Build square lattice with nearest and next-nearest neighbor edges
lattice = nk.graph.Square(L, max_neighbor_order=2)
hi = nk.hilbert.Spin(s=1 / 2, total_sz=0, N=lattice.n_nodes)
# Heisenberg with coupling J=1.0 for nearest neighbors
# and J=0.5 for next-nearest neighbors
H = nk.operator.Heisenberg(hilbert=hi, graph=lattice, J=[1.0, 0.5])

# Find an approximate ground state
machine = nk.models.GCNN(
    symmetries=lattice,
    parity=1,
    layers=4,
    features=4,
    dtype=complex,
)
sampler = nk.sampler.MetropolisExchange(
    hilbert=hi,
    n_chains=1024,
    graph=lattice,
    d_max=2,
)
opt = nk.optimizer.Sgd(learning_rate=0.02)
sr = nk.optimizer.SR(diag_shift=0.01)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=machine,
    n_samples=1024,
    n_discard_per_chain=0,
    chunk_size=4096,
)
gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)
gs.run(n_iter=200, out="ground_state")

data = json.load(open("ground_state.log"))
print(np.mean(data["Energy"]["Mean"]["real"][-20:]) / 400)
print(np.std(data["Energy"]["Mean"]["real"][-20:]) / 400)

# Find an excited state by specifying a nontrivial symmetry sector
saved_params = vstate.parameters
characters = lattice.space_group_builder().space_group_irreps(pi, pi)[0]
machine = nk.models.GCNN(
    symmetries=lattice,
    characters=characters,
    parity=-1,
    layers=4,
    features=4,
    dtype=complex,
)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=machine,
    n_samples=1024,
    n_discard_per_chain=0,
    chunk_size=4096,
)
# Reuse parameters converged for the ground state as a good initial guess
vstate.parameters = saved_params
gs = nk.driver.VMC(H, opt, variational_state=vstate, preconditioner=sr)
gs.run(n_iter=50, out="excited_state")

data = json.load(open("excited_state.log"))
print(np.mean(data["Energy"]["Mean"]["real"][-10:]) / 400)
print(np.std(data["Energy"]["Mean"]["real"][-10:]) / 400)
