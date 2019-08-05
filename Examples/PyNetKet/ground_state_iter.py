from __future__ import print_function
import netket as nk
import sys

from numpy.linalg import eigvalsh
import jax.experimental.optimizers as jaxopt

SEED = 3141592

# Constructing a 1d lattice
N = 64
g = nk.graph.Hypercube(length=N, n_dim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# Hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# Machine
ma = nk.machine.RbmSpinV2(hilbert=hi, alpha=2)
ma.init_random_parameters(seed=SEED, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisLocalV2(machine=ma, batch_size=32)
sa.seed(SEED)

mpi_rank = nk.MPI.rank()

if mpi_rank == 0:
    e0 = -1.27455 * N
    # e0 = nk.exact.lanczos_ed(ha).eigenvalues[0]
    print("  E0 = {: 10.4f}".format(e0))

FORMAT_STRING = (
    "{: 4} | {: 10.4f} ± {:.4f} | {:9.4f} | {:.4f} | {:.4f} | {: 7} | {:.2e} | {:.2e}"
)
HEADER_STRING = "step | E          ± σ(E)   | (ΔE)^2    | Rhat   | Accept | rank(S) | λ_min(S) | λ_max(S)"


def output(vmc, step):
    obs = vmc.get_observable_stats()
    if mpi_rank == 0:
        energy = obs["Energy"].mean
        sigma = obs["Energy"].error_of_mean
        variance = obs["Energy"].variance
        rhat = obs["Energy"].R

        # S = vmc._sr.last_covariance_matrix
        # w = eigvalsh(S)

        print(
            FORMAT_STRING.format(
                step,
                energy.real,
                sigma,
                variance,
                rhat,
                1,  # sa.acceptance,
                0,  # vmc._sr.last_rank,
                0,  # w.min(),
                0,  # w.max(),
            )
        )
        # Print output to the console immediately
        sys.stdout.flush()
        # Save current parameters to file
        ma.save("test.wf")


def run_vmc(steps, step_size, diag_shift, n_samples):
    opt = jaxopt.sgd(step_size)
    # opt = nk.optimizer.Sgd(step_size)

    sr = nk.optimizer.SR(use_iterative=True, diag_shift=diag_shift)
    sr.store_rank_enabled = False
    sr.store_covariance_matrix_enabled = False

    vmc = nk._driver.VmcDriver(
        hamiltonian=ha,
        machine=ma,
        sampler=sa,
        optimizer=opt,
        n_samples=n_samples,
        n_discard=min(n_samples // 10, 200),
        sr=sr,
    )

    if mpi_rank == 0:
        print(vmc)
        print(HEADER_STRING)

    for step in vmc.iter(steps, 1):
        output(vmc, step)


run_vmc(50, 0.02, 1e-3, 256)
run_vmc(50, 0.05, 1e-3, 1024)
run_vmc(50, 0.03, 1e-3, 2048)
run_vmc(50, 0.02, 1e-3, 4096)
run_vmc(100, 0.01, 1e-3, 16384)
run_vmc(100, 0.005, 1e-3, 32768)
