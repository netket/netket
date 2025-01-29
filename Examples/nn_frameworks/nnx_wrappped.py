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

import jax
import jax.numpy as jnp
from flax import nnx
import optax

import numpy as np


class RBM(nnx.Module):
    def __init__(
        self, N, alpha, rngs: nnx.Rngs, visible_bias: bool = True, param_dtype=complex
    ):
        self.linear = nnx.Linear(N, alpha * N, param_dtype=param_dtype, rngs=rngs)
        if visible_bias:
            self.visible_bias = nnx.Param(
                jax.random.uniform(rngs.params(), (N,), dtype=param_dtype)
            )
        else:
            self.visible_bias = False

    def __call__(self, x_in):
        y = nk.nn.log_cosh(self.linear(x_in))
        y = jnp.sum(y, axis=-1)
        if self.visible_bias is not None:
            # print(y)
            # print(self.visible_bias)
            # print("type", type(self.visible_bias))
            y = y + jnp.dot(x_in, self.visible_bias.raw_value)

        return y

    def another_fun(self, x_in):
        y = nk.nn.log_cosh(self.linear(x_in))
        return y


# 1D Lattice
L = 5
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
s = hi.numbers_to_states(np.arange(10))

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

xs = hi.random_state(jax.random.key(1), (3,))

# RBM Spin Machine
ma = RBM(
    hi.size, alpha=2, visible_bias=True, param_dtype=float, rngs=nnx.Rngs(0)
)  # eager initialization
# model = RBM(hi.size, alpha=2, visible_bias=True, param_dtype=float, rngs=nnx.Rngs(0))  # eager initialization
ma2 = nnx.bridge.to_linen(
    RBM, hi.size, alpha=2, visible_bias=True, param_dtype=float
)  # , rngs=nnx.Rngs(0))

r = jax.jit(lambda x: ma(x))(hi.all_states())

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer with a decreasing learning rate
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))

# SR
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
vs = nk.vqs.MCState(sa, ma2, n_samples=1008, n_discard_per_chain=10)

# Variational monte carlo driver with a variational state
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Run the optimization for 500 iterations
log = nk.logging.JsonLog("ciao")
gs.run(n_iter=500, out=log, timeit=True)
