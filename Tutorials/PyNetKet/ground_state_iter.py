from __future__ import print_function
from mpi4py import MPI
import netket as nk
import sys

SEED = 3141592

# Constructing a 1d lattice
g = nk.graph.Hypercube(length=20, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# Hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=SEED, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisLocal(machine=ma)
sa.seed(SEED)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Variational Monte Carlo
vmc = nk.gs.vmc.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.0,
    method='Sr')

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()

for i, st in enumerate(vmc.iter()):
    obs = dict(st.observables) # TODO: needs to be called on all MPI processes
    if mpi_rank == 0:
        print("step={}".format(i))
        print("acceptance={}".format(list(st.acceptance)))
        print("observables={}\n".format(obs))
        sys.stdout.flush()
    comm.barrier()
