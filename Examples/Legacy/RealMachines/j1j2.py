# Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.
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
from netket import legacy as nk

# Sigma^z*Sigma^z interactions
sigmaz = np.array([[1, 0], [0, -1]])
mszsz = np.kron(sigmaz, sigmaz)

# Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

# Couplings J1 and J2
J = [1, 0.5]

L = 20

mats = []
sites = []
for i in range(L):

    for d in [0, 1]:
        # \sum_i J*sigma^z(i)*sigma^z(i+d)
        mats.append((J[d] * mszsz).tolist())
        sites.append([i, (i + d + 1) % L])

        # \sum_i J*(sigma^x(i)*sigma^x(i+d) + sigma^y(i)*sigma^y(i+d))
        mats.append(((-1.0) ** (d + 1) * J[d] * exchange).tolist())
        sites.append([i, (i + d + 1) % L])

# Custom Graph
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=1 / 2, total_sz=0.0, N=g.n_nodes)

# Custom Hamiltonian operator
op = nk.operator.LocalOperator(hi)
for mat, site in zip(mats, sites):
    op += nk.operator.LocalOperator(hi, mat, site)

# Restricted Boltzmann Machine in the phase representation
ma = nk.machine.RbmSpinPhase(hi, alpha=1)
ma.init_random_parameters(seed=1234, sigma=0.1)

# Sampler
sa = nk.sampler.MetropolisExchange(machine=ma, graph=g.n_nodes)

# Optimizer
opt = nk.optimizer.Sgd(learning_rate=0.01)

# Variational Monte Carlo
gs = nk.variational.VMC(
    hamiltonian=op, sampler=sa, optimizer=opt, n_samples=1000, method="Sr"
)

gs.run(out="test", n_iter=10000)
