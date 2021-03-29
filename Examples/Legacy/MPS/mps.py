from netket import legacy as nk
import numpy as np
import jax
from jax.experimental.optimizers import adam as Adam

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2) ** L

ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

ma = nk.machine.MPSPeriodic(
    hi, g, bond_dim=4, diag=False, symperiod=None, dtype=complex
)
ma.jax_init_parameters(seed=1232)

# Jax Sampler
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=2)

# Using native Jax Optimizers under the hood
op = nk.optimizer.jax.Wrap(ma, Adam(0.01))
sr = nk.optimizer.SR(ma, diag_shift=0.1)

# Create the optimization driver
gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=1000, sr=sr)

# The first iteration is slower because of start-up jit times
gs.run(out="test", n_iter=1)

gs.run(out="test", n_iter=300)
