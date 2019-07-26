from __future__ import print_function
import netket as nk
import sys

from numpy.linalg import eigvalsh

SEED = 3141592

# Constructing a 1d lattice
N = 20
g = nk.graph.Hypercube(length=N, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# Hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# Machine
ma = nk.machine.RbmSpinSymm(hilbert=hi, alpha=4)
ma.init_random_parameters(seed=SEED, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisLocal(machine=ma)
sa.seed(SEED)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.1)

mpi_rank = nk.MPI.rank()

if mpi_rank == 0:
    e0 = -1.27455 * N
    print("  E0 = {: 10.4f}".format(e0))

FORMAT_STRING = (
    "{: 4} | {: 10.4f} ± {:.4f} | {:9.4f} ± {:.4f} | {:.4f} | {: 7} | {:.2e} | {:.2e}"
)
HEADER_STRING = "step | E          ± σ(E)   | (ΔE)^2    ± σ(...) | Accept | rank(S) | λ_min(S) | λ_max(S)"


def output(vmc, step):
    obs = vmc.get_observable_stats()
    if mpi_rank == 0:
        energy = obs["Energy"]["Mean"]
        sigma = obs["Energy"]["Sigma"]
        variance = obs["EnergyVariance"]["Mean"]
        vsigma = obs["EnergyVariance"]["Sigma"]

        S = vmc.last_S_matrix
        w = eigvalsh(S)

        print(
            FORMAT_STRING.format(
                step,
                energy,
                sigma,
                variance,
                vsigma,
                sa.acceptance[0],
                vmc.last_rank,
                w.min(),
                w.max(),
            )
        )
        # Print output to the console immediately
        sys.stdout.flush()
        # Save current parameters to file
        ma.save("test.wf")


def run_vmc(steps, step_size, diag_shift):
    op = nk.optimizer.Sgd(step_size)
    vmc = nk.variational.Vmc(
        hamiltonian=ha,
        sampler=sa,
        optimizer=op,
        n_samples=2000,
        diag_shift=diag_shift,
        method="Sr",
    )
    vmc.store_rank = True
    vmc.store_S_matrix = True

    if mpi_rank == 0:
        print(HEADER_STRING)

    for step in vmc.iter(steps):
        output(vmc, step)


run_vmc(50, 0.05, 1e-9)
run_vmc(50, 0.04, 1e-9)
run_vmc(50, 0.03, 1e-9)
run_vmc(50, 0.02, 1e-9)
run_vmc(100, 0.01, 1e-9)
run_vmc(100, 0.005, 1e-9)
