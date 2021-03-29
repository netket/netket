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
import networkx as nx

# defining a custom graph
# here we use networkx to generate a star graph
# and pass its edges list to NetKet custom graph
gnx = nx.star_graph(10)
g = nk.graph.CustomGraph(list(gnx.edges))

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

# RBM Spin Machine
ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.1)

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.1,
    method="Sr",
)

gs.run(out="test", n_iter=500)
