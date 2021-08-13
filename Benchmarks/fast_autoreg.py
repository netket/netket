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

import argparse
import time

import jax
import netket as nk
from jax import numpy as jnp

parser = argparse.ArgumentParser()
parser.add_argument("--L", type=int, default=10000)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--features", type=int, default=16)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--fast", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--dtype", type=str, default="float32", choices=["float32", "float64"]
)
args = parser.parse_args()

if args.fast:
    model_type = nk.models.FastARNNConv1D
else:
    model_type = nk.models.ARNNConv1D

if args.dtype == "float32":
    dtype = jnp.float32
elif args.dtype == "float64":
    dtype = jnp.float64
else:
    raise ValueError(f"Unknown dtype: {args.dtype}")

g = nk.graph.Hypercube(length=args.L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin Hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1)

# Autoregressive neural network
ma = model_type(
    hilbert=hi,
    layers=args.layers,
    features=args.features,
    kernel_size=args.kernel_size,
    dtype=dtype,
)

# Autoregressive direct sampling
sa = nk.sampler.ARDirectSampler(hi, n_chains=args.batch_size, dtype=dtype)

# Variational state
# With direct sampling, we don't need many samples in each step to form a
# Markov chain, and we don't need to discard samples
vs = nk.vqs.MCState(sa, ma, n_samples=args.batch_size)
assert vs.chain_length == 1
assert vs.n_discard_per_chain == 0

# n_parameters also takes masked parameters into account
# The number of non-masked parameters is about a half
print("n_parameters:", vs.n_parameters)

time_begin = time.time()
samples = vs.sample()
samples.block_until_ready()
time_spent = time.time() - time_begin
print("JIT time:", time_spent)

if args.profile:
    jax.profiler.start_trace("/tmp/tensorboard")

time_begin = time.time()
for i in range(3):
    if args.profile:
        with jax.profiler.StepTraceAnnotation("sample", step_num=i):
            samples = vs.sample()
    else:
        samples = vs.sample()
samples.block_until_ready()
time_spent = time.time() - time_begin

if args.profile:
    jax.profiler.stop_trace()

print("Sampling time:", time_spent)
