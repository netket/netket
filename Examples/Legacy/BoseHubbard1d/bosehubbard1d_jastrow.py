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

# 1D Periodic Lattice
g = nk.graph.Hypercube(length=12, n_dim=1, pbc=True)

# Boson Hilbert Space
hi = nk.hilbert.Fock(N=g.n_nodes, n_max=3, n_bosons=12)

# Bose Hubbard Hamiltonian
ha = nk.operator.BoseHubbard(U=4.0, hilbert=hi, graph=g)

# Jastrow Machine with Symmetry
ma = nk.machine.JastrowSymm(hilbert=hi, automorphisms=g)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)

# Stochastic gradient descent optimization
op = nk.optimizer.Sgd(ma, learning_rate=0.1)

# Variational Monte Carlo
vmc = nk.variational.VMC(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=5e-3,
    use_iterative=False,
    method="Sr",
)

vmc.run(n_iter=4000, out="test")
