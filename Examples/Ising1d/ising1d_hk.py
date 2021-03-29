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

import netket as nk
import haiku as hk
import jax.numpy as jnp
import numpy as np


class MyLinear(hk.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    def __call__(self, x):
        j, k = x.shape[-1], self.output_size
        w_init = hk.initializers.TruncatedNormal(1.0 / np.sqrt(j))
        w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.zeros)
        return jnp.dot(x, w) + b


def forward_fn(x):
    model = MyLinear(10)
    return jnp.squeeze(jnp.sum(model(x), axis=-1))


ma = hk.transform(forward_fn)

# 1D Lattice
L = 20  # 10

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.1)

# Variational monte carlo driver
gs = nk.VMC(ha, op, sa, ma, n_samples=1000, n_discard=50)

# Run the optimization for 300 iterations
gs.run(n_iter=300, out="test")
