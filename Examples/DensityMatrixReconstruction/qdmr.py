import netket as nk
import numpy as np
from generate_data import generate
import sys
import json

mpi_rank = nk.MPI.rank()

rg = nk.utils.RandomEngine(seed=1234)

nk.utils.seed(123)

# Generate and load the data
N = 8
hi, rotations, training_samples, training_bases, ha, psi = generate(
    N, n_basis=2 * N, n_shots=1000, seed=1234
)
# print(training_samples[0:10])

# Machine
ma = nk.machine.NdmSpinPhase(hilbert=hi, alpha=1, n_ancilla=1)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.ExactSampler(machine=ma.diagonal(), sample_size=16)

# Optimizer
op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)
# Quantum State Reconstruction
qst = nk.Qdmr(
    machine=ma,
    sampler=sa,
    optimizer=op,
    samples=training_samples,
    rotations=rotations,
    bases=training_bases,
    n_samples=1000,
    n_samples_data=1000,
    sr=sr,
)

qst.add_observable(ha, "Energy")
# qst.test_derivatives(epsilon=1e-5)

for step in qst.iter(50000, 50):
    obs = qst.get_observable_stats()
    if mpi_rank == 0:
        # print("step={}".format(step))
        # print("observables={}".format(obs))

        # # Compute fidelity with exact state
        rho_ma = ma.to_matrix()
        fidelity = np.abs(np.vdot(psi, np.dot(rho_ma, psi)))
        # print("fidelity={}".format(fidelity))

        # Compute NLL on training data
        nll = qst.nll(
            rotations=rotations,
            samples=training_samples,
            bases=training_bases,
            log_trace=ma.log_trace(),
        )
        # print("negative log likelihood={}".format(nll))

        # Print output to the console immediately
        print("Steps = %d\t" % step, end="")
        print("Fidelity = %.5f\t" % fidelity, end="")
        print("NLL = %.5f\t" % nll, end="")
        print("<H> = %.5f\t" % obs["Energy"].mean.real, end="")
        # print('(exact = -3.493959)',end='')
        print()

        sys.stdout.flush()


print()
