# Copyright 2025 The Netket Authors. - All Rights Reserved.
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
import netket.experimental as nkx
import optax
import jax

# 2D Lattice
g = nk.graph.Hypercube(length=5, n_dim=2, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian at the critical point
ha = nk.operator.IsingJax(hilbert=hi, graph=g, h=3.0)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, use_visible_bias=True, param_dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# The variational state
vs = nk.vqs.MCState(sa, ma, n_samples=1008, n_discard_per_chain=10)
vs.init_parameters(jax.nn.initializers.normal(stddev=0.01), seed=1234)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=optax.linear_schedule(0.01, 0.0001, 1000))

# Variational monte carlo driver
gs = nkx.driver.VMC_SR(
    ha, op, variational_state=vs, diag_shift=0.001, use_ntk=True, on_the_fly=False
)

# Create a JSON output file, and overwrite if file exists
logger = nk.logging.JsonLog("test", "w")

# Run the optimization
gs.run(n_iter=1000, out=logger)
