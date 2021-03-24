# Copyright 2020 The Netket Authors. - All Rights Reserved.
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

from netket import legacy as nk

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.nn.models.RBM(alpha=1)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=32)

# Optimizer
op = nk.optim.GradientDescent(learning_rate=0.1)

# Create the optimization driver
vs = nk.variational_states.ClassicalVariationalState(
    ma, sa, n_samples=1000, n_discard=100
)

gs = nk.Vmc(ha, op, variational_state=vs)

# Run the optimization for 300 iterations
gs.run(n_iter=2, out=None)

gs.run(n_iter=300, out=None)
