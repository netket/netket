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

from netket.operator import local_values

# 1D Lattice
g = nk.graph.Hypercube(length=40, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# Ising spin hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# Hilbert space of spins on the graph
py_hi = nk.hilbert.PySpin(s=0.5, graph=g)

# Ising spin hamiltonian
py_ha = nk.operator.PyIsing(h=1.0, hilbert=py_hi)

# RBM Spin Machine
ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=16)

n_samples = 1000
samples = np.zeros((n_samples, sa.sample_shape[0], sa.sample_shape[1]))
for i, sample in enumerate(sa.samples(n_samples)):
    samples[i] = sample


loc = np.empty(samples.shape[0:2], dtype=np.complex128)


def compute_locals(ha, n_times):
    for k in range(n_times):
        for i, sample in enumerate(samples):
            local_values(ha, ma, sample, out=loc[i])


compute_locals(ha, 1)
cProfile.run("compute_locals(ha,10)")


compute_locals(py_ha, 1)
cProfile.run("compute_locals(py_ha,10)")
