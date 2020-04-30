import netket as nk
import numpy as np
import time
import jax
from jax.config import config
from functools import partial
from jax.tree_util import tree_flatten, tree_unflatten, tree_map

# config.update("jax_log_compiles", 1)

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.PySpin(s=0.5, graph=g)

# Ising spin hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# RBM Spin Machine
alpha = 1
dtype = float


ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi, use_visible_bias=False, dtype=dtype)


j_ma = nk.machine.JaxRbm(hi, alpha, dtype=dtype)

# Metropolis Local Sampling
n_chains = 8

j_sa = nk.sampler.jax.MetropolisLocal(machine=j_ma, n_chains=n_chains)
sa = nk.sampler.MetropolisLocal(machine=j_ma, n_chains=n_chains)

n_samples = 1000 // n_chains


def j_bench(n_times, samples):
    for j in range(n_times):
        j_ma.init_random_parameters(sigma=0.01)
        out = j_ma.der_log(samples.reshape((-1, hi.size)))

    return out


def bench(n_times, samples):
    for j in range(n_times):
        ma.init_random_parameters(sigma=0.01)
        out = ma.der_log(samples.reshape((-1, hi.size)))

    return out


samples = j_sa.generate_samples(n_samples)
m = j_bench(1, samples)
t0 = time.time_ns()
samples = j_bench(300, samples)
tf = time.time_ns()
print("Jax der_log (dtype " + str(dtype) + ")")
print("time (s) ", (tf - t0) / 1.0e9)

samples = sa.generate_samples(n_samples)
m = bench(1, samples)
t0 = time.time_ns()
samples = bench(300, samples)
print("Numpy der_log (dtype " + str(dtype) + ")")
tf = time.time_ns()
print("time (s) ", (tf - t0) / 1.0e9)
