# Copyright 2018-2020 The Simons Foundation, Inc. - All Rights Reserved.
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
g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)

# Boson Hilbert Space
hi = nk.hilbert.Fock(N=g.n_nodes, n_max=3, n_bosons=8)

# Bose Hubbard Hamiltonian
ha = nk.operator.BoseHubbard(hilbert=hi, graph=g, U=4.0)

# RBM Machine with one-hot encoding, real parameters, and symmetries
ma = nk.machine.RbmMultiVal(hilbert=hi, alpha=1, dtype=float, automorphisms=g)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler using Hamiltonian moves, thus preserving the total number of particles
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha, batch_size=16)

# Stochastic gradient descent optimization
op = nk.optimizer.Sgd(ma, 0.05)

# Variational Monte Carlo
sr = nk.optimizer.SR(ma, diag_shift=0.1)
vmc = nk.Vmc(
    hamiltonian=ha, sampler=sa, optimizer=op, n_samples=4000, n_discard=0, sr=sr
)

vmc.run(n_iter=300, out="test")
