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

# 1D Periodic Lattice
g = nk.graph.Hypercube(length=12, n_dim=1, pbc=True)

# Boson Hilbert Space
hi = nk.hilbert.Boson(graph=g, n_max=3, n_bosons=12)

# Bose Hubbard Hamiltonian
ha = nk.operator.BoseHubbard(U=4.0, hilbert=hi)

# Jastrow Machine with Symmetry
ma = nk.machine.JastrowSymm(hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Sampler
sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)

# Stochastic gradient descent optimization
op = nk.optimizer.Sgd(learning_rate=0.1)

# Variational Monte Carlo
vmc = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=5e-3,
    use_iterative=False,
    method="Sr",
)

vmc.run(output_prefix="test", n_iter=4000)
