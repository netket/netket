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
g = nk.graph.Hypercube(length=40, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# Ising spin hamiltonian
ha = nk.operator.Ising(h=1.0, hilbert=hi)

# RBM Spin Machine
alpha = 4
ma = nk.machine.RbmSpin(alpha=alpha, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)


py_ma = nk.machine.PyRbm(alpha=alpha, hilbert=hi)
py_ma.parameters = ma.parameters

# Metropolis Local Sampling
batch_size = 16
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=batch_size)

n_samples = 1000
samples = np.zeros((n_samples, sa.sample_shape[0], sa.sample_shape[1]))
for i, sample in enumerate(sa.samples(n_samples)):
    samples[i] = sample


vals = np.zeros((n_samples, batch_size), dtype=np.complex128)


def log_val(n_times):
    for k in range(n_times):
        for i, sample in enumerate(samples):
            vals[i] = ma.log_val(sample)


def py_log_val(n_times):
    for k in range(n_times):
        for i, sample in enumerate(samples):
            py_ma.log_val(sample, out=vals[i])


cProfile.run("log_val(300)")
cProfile.run("py_log_val(300)")
