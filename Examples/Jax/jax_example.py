import netket as nk
import numpy as np


# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2) ** L

ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

alpha = 1
ma = nk.machine.JaxRbm(hi, alpha, dtype=float)
ma.init_random_parameters(seed=1232)

# Jax Sampler
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=2)

# Using Sgd
op = nk.optimizer.Sgd(ma, learning_rate=0.1)


# Create the optimization driver
gs = nk.Vmc(
    hamiltonian=ha, sampler=sa, optimizer=op, n_samples=1000, sr=None, n_discard=None
)

# The first iteration is slower because of start-up jit times
gs.run(out="test", n_iter=2)

gs.run(out="test", n_iter=300)
