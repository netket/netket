# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk
import numpy as np

# Sigma^z*Sigma^z interactions
sigmaz = np.array([[1, 0], [0, -1]])
mszsz = np.kron(sigmaz, sigmaz)

# Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

# Couplings J1 and J2
J = [1, 0.4]

L = 20

mats = []
sites = []
for i in range(L):

    for d in [0, 1]:
        # \sum_i J*sigma^z(i)*sigma^z(i+d)
        mats.append((J[d] * mszsz))
        sites.append([i, (i + d + 1) % L])

        # \sum_i J*(sigma^x(i)*sigma^x(i+d) + sigma^y(i)*sigma^y(i+d))
        mats.append(((-1.0) ** (d + 1) * J[d] * exchange))
        sites.append([i, (i + d + 1) % L])

# Custom Graph
g = nk.graph.Chain(length=L, pbc=True)

# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

# Custom Hamiltonian operator
ha = nk.operator.LocalOperator(hi)
for mat, site in zip(mats, sites):
    ha += nk.operator.LocalOperator(hi, mat, site)

# Restricted Boltzmann Machine
ma = nk.models.RBM(alpha=1, use_visible_bias=True, dtype=complex)

# Exchange Sampler randomly exchange up to next-to-nearest neighbours
sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=16, d_max=2)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)

# Stochastic reconfiguration
sr = nk.optimizer.SR(diag_shift=0.1)

# Variational Monte Carlo
vmc = nk.VMC(ha, op, sa, ma, n_samples=4000, n_discard=100)

vmc.run(n_iter=300, out="test")
