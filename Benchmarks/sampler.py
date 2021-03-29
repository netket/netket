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

import cProfile
import netket as nk
import numpy as np

# 1D Lattice
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# Ising spin hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# RBM Spin Machine
alpha = 1
ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)


py_ma = nk.machine.PyRbm(alpha=alpha, hilbert=hi)
py_ma.parameters = ma.parameters

# Metropolis Local Sampling
batch_size = 32
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=batch_size)
py_sa = nk.sampler.MetropolisLocal(machine=py_ma, n_chains=batch_size)

n_samples = 500
samples = np.zeros((n_samples, sa.sample_shape[0], sa.sample_shape[1]))


vals = np.zeros((n_samples, batch_size), dtype=np.complex128)


def bench(n_times, sampler):
    for k in range(n_times):
        for i, sample in enumerate(sampler.samples(n_samples)):
            samples[i] = sample


bench(1, py_sa)

cProfile.run("bench(30,sa)")
cProfile.run("bench(30,py_sa)")
