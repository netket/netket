# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To test this script on your local computer, you can use the following
# command:
# mpirun -np 2 python netket/Examples/multi_process.py
#
# This assumes that you have mpi4py installed, otherwise
# refer to jax's guide
# this same script can also be used on SLURM clusters

# Set this BEFORE importing netket. Or you can set it in your shell
import os

os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

import jax
import netket as nk
import numpy as np
import optax

# Initialize jax distributed. This must be done before any operation
# is performed on jax arrays.
print("\n---------------------------------------------")
print("Initializing JAX distributed using GLOO...", flush=True)

# This line tells jax to use GLOO to make different processes communicate
# when performing computations on CPU. If you are running on GPU, this line
# is not necessary.
# GLOO is very slow, so it's here just for demonstration purposes and should
# be replaced by MPITrampoline if you are using the CPU (though getting it working
# is complex).
jax.config.update("jax_cpu_collectives_implementation", "gloo")

# Initializes the jax distributed environment using mpi4py to detect the
# number of processes and the local rank. This requires mpi4py to be installed.
#
# Sometimes this does not work very well. The default is not to declare it, which
# will make jax default to checking with SLURM (if you are using slurm) or fail.
jax.distributed.initialize(cluster_detection_method="mpi4py")

default_string = f"r{jax.process_index()}/{jax.process_count()}: "
print(default_string, jax.devices(), flush=True)
print(default_string, jax.local_devices(), flush=True)
print("---------------------------------------------\n", flush=True)

# TO print something only on rank 0, do the following
if jax.process_index() == 0:
    print(default_string, "This is printed only on rank 0")

# 1D Lattice
L = 4
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, inverted_ordering=False)

# Ising spin hamiltonian
ha = nk.operator.IsingJax(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer with a decreasing learning rate
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))

# SR
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

# Variational monte carlo driver with a variational state
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# The logger is smart, and will accumulate data on every process, but will
# only save to disk on the master process (process #0)
log = nk.logging.JsonLog("test_sharding")

# Run the optimization for 500 iterations
gs.run(n_iter=50, out=log, timeit=True)

# Sharding will shard the samples. So if you look at them you will see that they have the shape
# (n_chains, n_samples_per_rank, hilbert.size), but the sharding is such that the
# n_chains are split among different ranks
print(default_string, "Samples shape:   ", vs.samples.shape, flush=True)
print(default_string, "Samples sharding:", vs.samples.sharding, flush=True)
# The sharding will be (#,1,1), which means that the first axis is sharded, but the other two are not.
# You cannot access/print vs.samples directly, as it is sharded and all data is not available on a single
# rank.
print(
    default_string,
    "Samples is_fully_addressable: ",
    vs.samples.is_fully_addressable,
    flush=True,
)
try:
    print(vs.samples)
except Exception as e:
    print(default_string, "Error accessing samples:", e, flush=True)

# to get the data from all samples on all ranks, you can do the following: make it replicated, meaning
# that you set the sharding to (1,1,1), which means that all data is available on all ranks.
samples_replicated = jax.lax.with_sharding_constraint(
    vs.samples, shardings=jax.sharding.PositionalSharding(jax.devices()).replicate()
)

print(
    default_string,
    "A quantity computed on a single rank, depending on all data: ",
    np.array(samples_replicated).sum(),
    flush=True,
)
