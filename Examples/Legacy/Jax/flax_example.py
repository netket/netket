from netket import legacy as nk
import flax
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np


# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5) ** L

ha = nk.operator.Ising(h=1.0, hilbert=hi, graph=g)

alpha = 1


class RBM(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=alpha * x.shape[-1], dtype=jax.numpy.float32)(x)
        x = jnp.log(jnp.cosh(x))
        return jnp.sum(x, axis=-1)


ma = nk.machine.Flax(hi, RBM())
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
