import netket as nk
import numpy as np
from generate_data import generate
import sys

mpi_rank = nk.MPI.rank()

rg = nk.utils.RandomEngine(seed=1234)

nk.utils.seed(123)

# Generate and load the data
N = 3
hi, rotations, training_samples, training_bases, ha, psi = generate(
    N, n_basis=2 * N, n_shots=500, seed=1234
)

# Machine
ma = nk.machine.NdmSpinPhase(hilbert=hi, alpha=1, beta=1)
ma.init_random_parameters(seed=1234, sigma=0.001)

# Sampler
sa = nk.sampler.MetropolisLocal(machine=ma.diagonal(), n_chains=4)

# Optimizer
op = nk.optimizer.Sgd(0.0001)

# Quantum State Reconstruction
qst = nk.Qdmr(
    machine=ma,
    sampler=sa,
    optimizer=op,
    samples=training_samples,
    rotations=rotations,
    bases=training_bases,
    n_samples=500,
    n_samples_data=500,
    sr=None,
)


qst.add_observable(ha, "Energy")


for step in qst.iter(500, 50):
    obs = qst.get_observable_stats()
    if mpi_rank == 0:
        print("step={}".format(step))
        print("observables={}".format(obs))

        # # Compute fidelity with exact state
        rho_ma = ma.to_matrix()
        #
        fidelity = np.abs(np.vdot(psi, np.dot(rho_ma, psi)))
        print("fidelity={}".format(fidelity))

        # Compute NLL on training data
        nll = qst.nll(
            rotations=rotations,
            samples=training_samples,
            bases=training_bases,
            log_trace=ma.log_trace(),
        )
        print("negative log likelihood={}".format(nll))

        # Print output to the console immediately
        sys.stdout.flush()
