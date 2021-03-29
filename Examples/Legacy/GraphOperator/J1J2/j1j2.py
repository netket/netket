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
import numpy as np
import networkx as nx
from netket import legacy as nk

# Sigma^z*Sigma^z interactions
sigmaz = [[1, 0], [0, -1]]
mszsz = np.kron(sigmaz, sigmaz)

# Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

# Couplings J1 and J2
J = [1, 0.4]
L = 20

pars = {}

# Define bond operators, labels, and couplings
bond_operator = [
    (J[0] * mszsz).tolist(),
    (J[1] * mszsz).tolist(),
    (-J[0] * exchange).tolist(),
    (J[1] * exchange).tolist(),
]

bond_color = [1, 2, 1, 2]

# Define custom graph
G = nx.Graph()
for i in range(L):
    G.add_edge(i, (i + 1) % L, color=1)
    G.add_edge(i, (i + 2) % L, color=2)

edge_colors = [[u, v, G[u][v]["color"]] for u, v in G.edges]

# Custom Graph
g = nk.graph.CustomGraph(edge_colors)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=1 / 2, total_sz=0.0, N=g.n_nodes)

# Custom Hamiltonian operator
op = nk.operator.GraphOperator(hi, bondops=bond_operator, bondops_colors=bond_color)

# Restricted Boltzmann Machine
ma = nk.machine.RbmSpin(hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisHamiltonianPt(machine=ma, hamiltonian=op, n_replicas=16)

# Optimizer
opt = nk.optimizer.Sgd(ma, learning_rate=0.01)

# Variational Monte Carlo
gs = nk.variational.Vmc(
    hamiltonian=op,
    sampler=sa,
    optimizer=opt,
    n_samples=1000,
    use_iterative=True,
    method="Sr",
)

gs.run(out="test", n_iter=10000)
