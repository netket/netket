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

rg = nk.utils.RandomEngine(seed=1234)

# 1D Lattice
L = 6
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)


# Defining the Ising hamiltonian (with sign problem here)
# Using local operators
sx = [[0, 1], [1, 0]]
sy = [[0, -1j], [1j, 0]]
sz = [[1, 0], [0, -1]]

sigmam = [[0, 0], [1, 0]]

ha = nk.operator.LocalOperator(hi)

for i in range(L):
    ha += nk.operator.LocalOperator(hi, sx, [i])
    ha += nk.operator.LocalOperator(hi, np.kron(sz, sz), [i, (i + 1) % L])


# Create the lindbladian with no jump operators
lind = nk.operator.LocalLindbladian(ha)

# Add a sigmam jump operator on each site
for i in range(L):
    j_op = nk.operator.LocalOperator(hi, sigmam, [i])
    lind.add_jump_op(j_op)


# RBM Spin Machine
ma = nk.machine.NdmSpinPhase(hilbert=hi, alpha=1, beta=2)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma)

# Optimizer
op = nk.optimizer.AdaDelta()


# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=lind,
    sampler=sa,
    optimizer=op,
    n_samples=300,
    diag_shift=0.1,
    use_iterative=True,
    method="Sr",
)


gs.run(output_prefix="ttest", n_iter=300)
