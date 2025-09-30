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
# djaxrun -np 2 python netket/Examples/multi_process.py
#
# on slurm clusters, srun should tak care of it.
#
# This assumes that you have mpi4py installed, otherwise
# refer to jax's guide
# this same script can also be used on SLURM clusters

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import numpy as np
import optax


# Initialize jax distributed. This must be done before any operation
# is performed on jax arrays and BEFORE you import netket.
print("\n---------------------------------------------")
print("Initializing JAX distributed...", flush=True)

# Initializes the jax distributed environment using env variable detection
# to detect the number of processes and the local rank
#
# djaxrun pretends you are running inside of slurm to make it work
jax.distributed.initialize()


import netket as nk

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

# Variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

# Variational monte carlo driver with a variational state
gs = nk.driver.VMC_SR(ha, op, variational_state=vs, diag_shift=0.01)

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
print(
    "Trying to `print(vs.samples)`. This will fail because it's not fully addressable."
)
try:
    print(vs.samples)
except Exception as e:
    print(default_string, "Error accessing samples:", e, flush=True)

# to get the data from all samples on all ranks, you can do the following: make it replicated, meaning
# that you set the sharding to (1,1,1), which means that all data is available on all ranks.
samples_replicated = jax.lax.with_sharding_constraint(
    vs.samples,
    shardings=jax.sharding.NamedSharding(jax.sharding.get_abstract_mesh(), P()),
)

print(
    default_string,
    "A quantity computed on a single rank, depending on all data: ",
    np.array(samples_replicated).sum(),
    flush=True,
)


# The following code will cause a deadlock:
print("Running a code that will deadlock (use Ctrl-C to exit)")
print("Showing this so you know what you shan't do in 'da zukumpft")
counter = jax.lax.with_sharding_constraint(
    jnp.zeros((jax.device_count(),)),
    shardings=jax.sharding.NamedSharding(jax.sharding.get_abstract_mesh(), P("S")),
)

# This code deadlocks because in jax all `jnp/array` operations should be
# performed on all processes, otherwise you are likely to get deadlocks.
# This is because jax array operations might hide global communication.
#
# In the example below, we have an array of integers, where each element is stored on
# a different process.
# The addition is performed on each local process and it is fine but
for i in range(10):
    print(f"Performing iteration {i} in the loop")
    counter = counter + i
    if jax.process_index() == 0:
        # To compute this .sum() the two processes must communicate.
        # This is like a MPI.ALLREDUCE operation.
        # when you run this .sum(), process 0 will send data to process 1, and will wait from
        # process 1 to get his data.
        # As process 1 never executes .sum(), he will never send this data and will end up
        # waiting forever.
        print(f"Executing counter_total = counter.sum()`")
        counter_total = counter.sum()
        # this line below will possibly execute because jax is 'lazy' and did not really
        # do the operation above, but still advanced
        print(
            "actually waiting until `counter.sum()` is finished [this will never finish]"
        )
        jax.block_until_ready(counter_total)
        # the line below will never execute
        print("printing counter_total")
        print(counter_total)

# A good rule of thumb: never use `jax.process_index()` to gate logic with jax arrays.
# Outside of `jax.process_index() ==0` put conversions to numpy array.
# Within jax.process_index() == 0` only operate on numpy arrays, not jax arrays.
# That way, you should be able to avoid deadlocks.
