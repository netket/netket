# Copyright 2026 The NetKet Authors - All rights reserved.
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

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

import netket as nk


class RBM(nn.Module):
    N: int
    alpha: int
    visible_bias: bool = True
    param_dtype: type = complex

    def setup(self):
        self.linear = nn.Dense(self.alpha * self.N, param_dtype=self.param_dtype)
        if self.visible_bias:
            self.bias = self.param(
                "visible_bias",
                nn.initializers.zeros_init(),
                (self.N,),
                self.param_dtype,
            )
        else:
            self.bias = None

    def __call__(self, x_in):
        y = self.hidden_layer(x_in)
        y = jnp.sum(y, axis=-1)
        if self.bias is not None:
            y = y + jnp.dot(x_in, self.bias)

        return y

    def hidden_layer(self, x_in):
        return nk.nn.activation.log_cosh(self.linear(x_in))


# 1D Lattice
L = 10
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

model = RBM(hi.size, alpha=2, visible_bias=True, param_dtype=float)
variables = model.init(jax.random.key(0), hi.random_state(jax.random.key(1), (1,)))
bound_model = model.bind(variables)

# A bound linen module already carries its parameters and can be called directly.
xs = hi.random_state(jax.random.key(2), (3,))
bound_model(xs)
bound_model.hidden_layer(xs)

sa = nk.sampler.MetropolisLocal(hi, n_chains=16)
vs = nk.vqs.MCState(sa, bound_model, n_samples=1008, n_discard_per_chain=10)

print(jax.tree.map(lambda x: (x.shape, x.dtype), vs.variables))
print(vs.model(xs).shape)
print(vs.model.hidden_layer(xs).shape)


# NetKet unbinds the model internally, so the same linen-compatible path works here too.
@partial(jax.jit, static_argnames="linen_model")
def myfun_linen(linen_model, variables, xs):
    return linen_model.apply(variables, xs, method=linen_model.hidden_layer)


myfun_linen(vs._model, vs.variables, xs)

op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.1, 0.0001, 500))
sr = nk.optimizer.SR(diag_shift=0.01)
gs = nk.driver.VMC(ha, op, variational_state=vs, preconditioner=sr)
log = nk.logging.RuntimeLog()
gs.run(n_iter=500, out=log, timeit=True)
