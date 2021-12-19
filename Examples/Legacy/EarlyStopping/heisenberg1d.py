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

# TODO Help, nk.callbacks.EarlyStopping raises AttributeError: module 'netket' has no attribute 'callbacks'

# 1D Lattice
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
# with total Sz equal to 0
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Heisenberg hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)

# Symmetric RBM Spin Machine
ma = nk.machine.RbmSpin(alpha=1, hilbert=hi, automorphisms=g)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Exchange Sampling
# Notice that this sampler exchanges two neighboring sites
# thus preservers the total magnetization
sa = nk.sampler.MetropolisExchange(machine=ma)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.05)

# Stochastic Reconfifugration
sr = nk.optimizer.SR(ma, diag_shift=0.1)

# Early Stopping
es = nk.callbacks.EarlyStopping(patience=10)

# Variational Monte Carlo
gs = nk.VMC(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    n_discard_per_chain=2,
    sr=sr,
)

gs.run(out="test", n_iter=300, callback=es)
