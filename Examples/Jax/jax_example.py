import netket as nk
import numpy as np
import jax
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

ha = nk.operator.Ising(h=1.0, hilbert=hi)


def initializer(rng, shape):
    return np.random.normal(scale=0.1, size=shape)


def logcosh(x):
    x = jax.numpy.abs(x)
    return x + jax.numpy.logaddexp(-2.0 * x, 0) - jax.numpy.log(2.0)


LogCoshLayer = jax.experimental.stax.elementwise(logcosh)
alpha = 4
ma = nk.machine.Jax(
    hi,
    jax.experimental.stax.serial(
        jax.experimental.stax.Dense(alpha * L, initializer, initializer),
        LogCoshLayer,
        jax.experimental.stax.Dense(1, initializer, initializer),
    ),
    dtype=complex,
)

sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=32)

# Optimizer
op = nk.optimizer.Sgd(0.1)

# Stochastic reconfiguration
sr = nk.optimizer.SR(diag_shift=0.1, use_iterative=True)

# Variational Monte Carlo
gs = nk.Vmc(
    hamiltonian=ha, sampler=sa, optimizer=op, n_samples=1000, n_discard=0, sr=sr
)

gs.run(output_prefix="test", n_iter=300)
