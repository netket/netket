import netket as nk
import numpy as np
from basis_generators import BasisGeneratorFull
from generate_data import generate

import netket.experimental as nkx

import matplotlib.pyplot as plt


def infidelity(psi1, psi2):
    return 1 - np.abs(np.conj(psi1) @ psi2) ** 2


# parameters
# data
N = 3
pbc = False
n_shots = 1000
# model
alpha = 1
n_chains = 16
n_samples = 1024
n_discard = 128
# training
batch_size = 100
lr = 0.01
n_iter = 1000
log_fname = "log"
# plot
plot = False
plot_fname = "ising_1d_qsr.png"

basis_gen = BasisGeneratorFull(N)

# generate measurement data
hi, rotations, training_samples, ha, psi = generate(
    N, n_basis=len(basis_gen), n_shots=n_shots, basis_generator=basis_gen
)
E0 = np.real(np.conj(psi) @ ha.to_dense() @ psi)

# define model
ma = nk.models.RBM(alpha=alpha, param_dtype=complex)
sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
op = nk.optimizer.Adam(learning_rate=lr)
vs = nk.vqs.MCState(sa, ma, n_samples=n_samples, n_discard_per_chain=n_discard)

# define driver
qsr = nkx.QSR(
    training_data=(training_samples, rotations),
    training_batch_size=batch_size,
    optimizer=op,
    variational_state=vs,
)


# callback
def callback(step, logvals, driver):
    state = driver.state
    psi_pred = state.to_array(normalize=True)
    logvals["Infidelity"] = infidelity(psi, psi_pred)
    logvals["KL"] = driver.KL(psi, n_shots=n_shots)
    logvals["KL_whole"] = driver.KL_whole_training_set(psi, n_shots=n_shots)
    logvals["KL_exact"] = driver.KL_exact(psi, n_shots=n_shots)
    logvals["Energy"] = np.abs(
        np.real(np.conj(psi_pred) @ ha.to_dense() @ psi_pred) - E0
    )
    return True


# logger
logger = nk.logging.JsonLog(log_fname)

# training
qsr.run(n_iter=n_iter, callback=callback, out=logger)

# plot results
results = logger.data
fig = plt.figure(figsize=(8, 8))
iters = results["Infidelity"].iters
infid = results["Infidelity"].values
energy = results["Energy"].values
KL = results["KL"].values
KL_whole = results["KL_whole"].values
KL_exact = results["KL_exact"].values
plt.plot(iters, infid, color="red", label="Infidelity")
plt.plot(iters, energy, color="red", label="Energy", linestyle="--")
plt.plot(iters, KL, color="blue", label="KL", alpha=0.2)
plt.plot(iters, KL_whole, color="green", label="KL whole")
plt.plot(iters, KL_exact, color="orange", label="KL exact")
plt.xlabel("Iteration")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig(plot_fname)
if plot:
    plt.show()
