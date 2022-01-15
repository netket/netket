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
import torch
import numpy as np

# 1D Lattice
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)


input_size = hi.size
alpha = 1

model = torch.nn.Sequential(
    torch.nn.Linear(input_size, alpha * input_size),
    torch.nn.ReLU(),
    torch.nn.Linear(alpha * input_size, 2),
    torch.nn.ReLU(),
)

ma = nk.machine.Torch(model, hilbert=hi)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=8)

# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.1)

# Stochastic reconfiguration
sr = nk.optimizer.SR(ma, diag_shift=0.1, use_iterative=True)

# Driver
gs = nk.VMC(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=500, sr=sr)

gs.run(n_iter=300, out="test")
