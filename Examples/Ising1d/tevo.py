import netket as nk
import numpy as np
from netket.dynamics import TimeEvolution

import netket as nk

# 1D Lattice
g = nk.graph.Hypercube(length=3, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi, graph=g)

Sx = sum([nk.operator.spin.sigmax(hi, i) for i in range(g.n_nodes)])

# RBM Spin Machine
ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(ma, n_chains=32)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.1)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma, diag_shift=0.1)

# Create the optimization driver
gs = nk.Vmc(ha, sampler=sa, optimizer=op, n_samples=1000, sr=sr)

# Run the optimization for 300 iterations
gs.run(n_iter=300, out="test")

# Now that we have the ground state, change the shift to a small value
sr = nk.optimizer.SR(ma, diag_shift=0.001, sparse_maxiter=1000)

tevo = TimeEvolution(
    ha,
    sampler=sa,
    sr=sr,
    n_samples=1000,
)
tevo.solver("rk45", dt=0.01, adaptive=False)

obs = {"Sx": Sx}

tevo.run(2.0, out="test_tevo", obs=obs)
