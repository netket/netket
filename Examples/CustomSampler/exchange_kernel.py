import netket as nk
import numpy as np

# 1D Lattice
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
# with total Sz equal to 0
hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)

# Heisenberg hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)

# Symmetric RBM Spin Machine
ma = nk.machine.RbmSpinV2(alpha=1, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Defining a custom kernel for MetropolisHastings
# Notice that this sampler exchanges two random sites
# thus preservers the total magnetization
def exchange_kernel(v, vnew, loprobcorr, n):

    vnew[:, :] = v[:, :]
    loprobcorr[:] = 0.0

    rands = np.random.randint(v.shape[1], size=(v.shape[0], 2))

    for i in range(v.shape[0]):
        iss = rands[i, 0]
        jss = rands[i, 1]

        vnew[i, iss], vnew[i, jss] = vnew[i, jss], vnew[i, iss]


sa = nk.sampler.MetropolisHastings(ma, exchange_kernel, batch_size=16, sweep_size=20)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.1,
    method="Sr",
)

gs.run(output_prefix="test", n_iter=300)
