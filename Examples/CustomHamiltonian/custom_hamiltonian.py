# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

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

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)


# Defining the Ising hamiltonian (with sign problem here)
# Using local operators
sx = [[0, 1], [1, 0]]
sz = [[1, 0], [0, -1]]

ha = nk.operator.LocalOperator(hi)

for i in range(L):
    ha += nk.operator.LocalOperator(hi, sx, [i])
    ha += nk.operator.LocalOperator(hi, np.kron(sz, sz), [i, (i + 1) % L])


# RBM Spin Machine
ma = nk.machine.RbmSpinPhase(alpha=1, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
op = nk.optimizer.AdaDelta()

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=300,
    diag_shift=0.1,
    use_iterative=True,
    method="Sr",
)

gs.run(output_prefix="test", n_iter=3000)
