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
import numpy as np

# 1D Lattice
l = 20
g = nk.graph.Hypercube(length=l, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
# with total Sz equal to 0
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Heisenberg hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)

# Symmetric RBM Spin Machine
ma = nk.machine.RbmSpinSymm(alpha=1, hilbert=hi, automorphisms=g)
ma.init_random_parameters(seed=1234, sigma=0.01)


# defining the custom sampler
# here we use two types of moves : 2-spin exchange between two random neighboring sites,
# and 4-spin exchanges between 4 random neighboring sites
# note that each line and column have to add up to 1.0 (stochastic matrices)

one_exchange_operator = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
two_exchange_operator = np.kron(one_exchange_operator, one_exchange_operator)


ops = [one_exchange_operator] * l + [two_exchange_operator] * l

# Single exchange operator acts on pairs of neighboring sites
acting_on = [[i, (i + 1) % l] for i in range(l)]

# Double exchange operator acts on cluster of 4 neighboring sites
acting_on += [[i, (i + 1) % l, (i + 2) % l, (i + 3) % l] for i in range(l)]


move_op = nk.operator.LocalOperator(hilbert=hi, operators=ops, acting_on=acting_on)

sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.05)

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.1,
    method="Sr",
)

gs.run(n_iter=300, out="test")
