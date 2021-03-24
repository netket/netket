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
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
# with total Sz equal to 0
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Heisenberg hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)

# Symmetric RBM Spin Machine
ma = nk.machine.JastrowSymm(hilbert=hi, automorphisms=g, dtype=float)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Exchange Sampling
# Notice that this sampler exchanges two neighboring sites
# thus preservers the total magnetization
sa = nk.sampler.MetropolisExchange(machine=ma)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.05)

# Stochastic reconfiguration
gs = nk.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    sr=nk.optimizer.SR(diag_shift=0.1, lsq_solver="QR"),
)

gs.run(out="test", n_iter=300)
