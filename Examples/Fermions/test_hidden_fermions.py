import netket as nk
from netket.models.hidden_fermion_determinant import HiddenFermionDeterminant
import jax 
from netket.operator.fermion import destroy as c
from netket.operator.fermion import create as cdag
from netket.operator.fermion import number as nc
from scipy.sparse.linalg import eigsh
from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd


seed = jax.random.PRNGKey(123456789)
hi = nk.hilbert.SpinOrbitalFermions(n_orbitals=4, s = 1/2, n_fermions_per_spin=(2,2))

U = 2.0
t = 1.0
H = 0.0

graph = nk.graph.Square(length=2, pbc=True)

for i, j in graph.edges():
    H += -t * (cdag(hi, i, sz=1) * c(hi, j, sz=1) + cdag(hi, j, sz=1) * c(hi, i, sz=1))
    H += -t * (cdag(hi, i, sz=-1) * c(hi, j, sz=-1) + cdag(hi, j, sz=-1) * c(hi, i, sz=-1))

for i in range(graph.n_nodes):
    H += U * nc(hi, i, sz=1) * nc(hi, i, sz=-1)

H = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(H)
model = HiddenFermionDeterminant(hi, n_hidden_fermions=1, hidden_unit_density=1)

sa = nk.sampler.MetropolisFermionHop(hi, graph=graph)
vstate = nk.vqs.MCState(sa, model, n_samples=128, n_discard_per_chain=8, seed=seed)
op = nk.optimizer.Sgd(learning_rate=0.05)
preconditioner = nk.optimizer.SR(diag_shift=0.01)
gs = nk.VMC(H, op, variational_state=vstate, preconditioner=preconditioner)

log = nk.logging.RuntimeLog()
gs.run(n_iter=500, out=log)

H_sp = H.to_sparse()
eig_vals, eig_vecs = eigsh(H_sp, k=2, which="SA")
E_gs = eig_vals[0]
print("Exact ground state energy:", E_gs)
