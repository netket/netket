import netket as nk
import numpy as np
import time


# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.PySpin(s=0.5, graph=g)

# Ising spin hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# RBM Spin Machine
alpha = 1
dtype = complex
ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, dtype=dtype)
ma.init_random_parameters(seed=1234, sigma=0.01)


j_ma = nk.machine.JaxRbm(alpha=alpha, hilbert=hi, dtype=dtype)
j_ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Local Sampling
n_chains = 2


j_sa = nk.sampler.jax.MetropolisLocal(machine=j_ma, n_chains=n_chains)
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=n_chains)

n_samples = 1000 // n_chains


def bench(n_times, sampler):
    for i in range(n_times):
        ma.init_random_parameters(seed=1234, sigma=0.01)
        samples = sampler.generate_samples(n_samples)
    return samples


def j_bench(n_times, sampler):
    for i in range(n_times):
        j_ma.init_random_parameters(seed=1234, sigma=0.01)
        samples = sampler.generate_samples(n_samples)
        samples.block_until_ready()
    return samples


samples = j_bench(1, j_sa)
samples.block_until_ready()
t0 = time.time_ns()
samples = j_bench(300, j_sa)
print("Jax sampler (dtype " + str(dtype) + ")")
tf = time.time_ns()
print("time (s) ", (tf - t0) / 1.0e9)


samples = bench(1, sa)
t0 = time.time_ns()
samples = bench(300, sa)
print("Numpy sampler (dtype " + str(dtype) + ")")
tf = time.time_ns()
print("time (s) ", (tf - t0) / 1.0e9)
