import netket as nk
import numpy as np
import jax
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh

# 1D Lattice
L = 4
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

ha = nk.operator.Ising(h=1.0, hilbert=hi)


def initializer(rng, shape):
    return np.random.normal(scale=0.1, size=shape)


ma = nk.machine.Jax(
    hi,
    jax.experimental.stax.serial(
        jax.experimental.stax.Dense(1 * L, initializer, initializer),
        jax.experimental.stax.Tanh,
        jax.experimental.stax.Dense(1, initializer, initializer)
    ),
    dtype=complex
)

sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.03)

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=100,
    diag_shift=0.01,
    method="Sr",
    use_iterative=True
)

gs.run(output_prefix="test", n_iter=100, show_progress=True)
