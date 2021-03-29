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
import numpy as np

sigmax = [[0, 1], [1, 0]]
sigmaz = [[1, 0], [0, -1]]

mszsz = (np.kron(sigmaz, sigmaz)).tolist()

# Notice that the Transverse-Field Ising model as defined here has sign problem
L = 20
site_operator = [sigmax]
bond_operator = [mszsz]

# Hypercube
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Custom Hilbert Space
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Graph Operator
op = nk.operator.GraphOperator(hi, siteops=site_operator, bondops=bond_operator)

# Restricted Boltzmann Machine
ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Local Metropolis Sampling
sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
opt = nk.optimizer.AdaMax(ma)

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=op,
    sampler=sa,
    optimizer=opt,
    n_samples=1000,
    diag_shift=0.1,
    method="Gd",
)

gs.run(out="test", n_iter=30000)
