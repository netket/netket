from __future__ import print_function
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
vmc = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.0,
    method="Sr",
)

mpi_rank = nk.MPI.rank()

for step in vmc.iter(300):
    obs = vmc.get_observable_stats()
    if mpi_rank == 0:
        print("step={}".format(step))
        print("acceptance={}".format(list(sa.acceptance)))
        print("observables={}\n".format(obs))
        # Print output to the console immediately
        sys.stdout.flush()
        # Save current parameters to file
        ma.save("test.wf")

    comm.barrier()
