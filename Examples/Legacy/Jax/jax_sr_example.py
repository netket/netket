from netket import legacy as nk
import jax
from jax.experimental.optimizers import sgd as JaxSgd

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

alpha = 1
ma = nk.machine.JaxRbm(hi, alpha, dtype=complex)
ma.init_random_parameters(seed=1232)

# Jax Sampler
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=2)

# Using Sgd
op = nk.optimizer.Sgd(ma, 0.05)


# Stochastic Reconfiguration
sr = nk.optimizer.SR(ma, diag_shift=0.1)

# Create the optimization driver
gs = nk.Vmc(
    hamiltonian=ha, sampler=sa, optimizer=op, n_samples=1000, sr=sr, n_discard=2
)

# The first iteration is slower because of start-up jit times
gs.run(out="test", n_iter=2)

gs.run(out="test", n_iter=300)
