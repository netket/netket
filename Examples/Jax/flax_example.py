import netket as nk
import flax
import jax


# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

ha = nk.operator.Ising(h=1.0, hilbert=hi)

alpha = 1


class flaxrbm(flax.nn.Module):
    def apply(self, x):
        x = flax.nn.Dense(x, features=alpha)
        x = flax.nn.log_sigmoid(x)
        return jax.numpy.sum(x, axis=-1)


ma = nk.machine.Jax(hi, flaxrbm, dtype=complex)
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

# gs.run(out="test", n_iter=300)
