# Copyright 2025 The NetKet Authors - All rights reserved.
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

from functools import partial

import netket as nk

import jax
import jax.numpy as jnp
from flax import nnx
import optax


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
        y = self.hidden_layer(x_in)
        y = jnp.sum(y, axis=-1)
        if self.visible_bias is not None:
            y = y + jnp.dot(x_in, self.visible_bias.raw_value)

        return y

    def hidden_layer(self, x_in):
        # You can define subfunctions, and you can use them at any point in time
        # contrary to what happens with linen
        y = nk.nn.log_cosh(self.linear(x_in))
        return y


# 1D Lattice
L = 10
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# Note that contrary to Flax.linen, when you construct the model you must
# supply an RNG key, and the model is initialized eagerly, and will contain the
# parameters!
ma = RBM(hi.size, alpha=2, visible_bias=True, param_dtype=float, rngs=nnx.Rngs(0))

# You can already use the network like the following:
xs = hi.random_state(jax.random.key(1), (3,))
ma(xs)
# or
ma.hidden_layer(xs)

# Build a variational state
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)

# You can get out the variational state parameters as usual
print(jax.tree.map(lambda x: (x.shape, x.dtype), vs.variables))
# You can also extract the model again
print(vs.model(xs).shape)
print(vs.model.hidden_layer(xs).shape)


# And to use it in your own code you can either do
@partial(jax.jit, static_argnames="graphdef")
def myfun(graphdef, variables, xs):
    model = nnx.merge(graphdef, variables)
    return model(xs)


graphdef, variables = nnx.split(vs.model)
myfun(graphdef, variables, xs)


# However NetKet internally wraps the model into a linen-like model stored in vs._model
# If you do something like that, your code will work for nnx and linen
@partial(jax.jit, static_argnames="linen_model")
def myfun_linen(linen_model, variables, xs):
    return linen_model.apply(variables, xs, method=linen_model.hidden_layer)


myfun_linen(vs._model, vs.variables, xs)

# train
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))
sr = nk.optimizer.SR(diag_shift=0.01)
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)
log = nk.logging.RuntimeLog()
gs.run(n_iter=500, out=log, timeit=True)
