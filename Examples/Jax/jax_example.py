import netket as nk
import numpy as np
import jax
import cProfile
from jax.config import config

# config.update("jax_log_compiles", 1)
# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.PySpin(s=0.5, graph=g)

ha = nk.operator.Ising(h=1.0, hilbert=hi)

alpha = 1
ma = nk.machine.JaxRbm(hi, alpha, dtype=complex)
ma.init_random_parameters(sigma=0.01, seed=1232)


# Jax Sampler
sa = nk.sampler.JaxMetropolisLocal(machine=ma, n_chains=16)

# Optimizer
op = nk.optimizer.Sgd(0.1)

# Stochastic reconfiguration
sr = nk.optimizer.SR(diag_shift=0.1, use_iterative=True)

# Variational Monte Carlo
gs = nk.Vmc(
    hamiltonian=ha, sampler=sa, optimizer=op, n_samples=1000, sr=sr, n_discard=0
)

# The first iteration is slower because of start-up jit times
gs.run(out="test", n_iter=1)

gs.run(n_iter=300, out="test")
