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
import numpy as np
from netket.operator.spin import sigmax, sigmay, sigmaz, sigmam, sigmap
import jax
from netket._dynamics import TimeEvolution

rg = nk.random.seed(seed=1234)

# 1D Lattice
L = 10
gp = 0.3
Vp = 2.0

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

# The hamiltonian
ha = nk.operator.LocalOperator(hi)

# List of dissipative jump operators
j_ops = []

# Observables
obs_sx = nk.operator.LocalOperator(hi)
obs_sy = nk.operator.LocalOperator(hi)
obs_sz = nk.operator.LocalOperator(hi)

for i in range(L):
    ha += (gp / 2.0) * sigmax(hi, i)
    ha += (Vp / 4.0) * sigmaz(hi, i) * sigmaz(hi, (i + 1) % L)
    j_ops.append(sigmam(hi, i))
    obs_sx += sigmax(hi, i)
    obs_sy += sigmay(hi, i)
    obs_sz += sigmaz(hi, i)


#  Create the liouvillian
lind = nk.operator.LocalLiouvillian(ha, j_ops)

# RBM Spin Machine
ma = nk.machine.density_matrix.NdmSpinPhase(hilbert=hi, alpha=1, beta=1)
ma.init_random_parameters(seed=1234, sigma=0.001)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(machine=ma, n_chains=8)
sa_obs = nk.sampler.MetropolisLocal(machine=ma.diagonal(), n_chains=8)

sr = nk.optimizer.SR(
    ma, diag_shift=0.001, use_iterative=True, sparse_tol=1e-6, sparse_maxiter=1000
)

tevo = TimeEvolution(
    lind, sa, sr=sr, n_samples=2000, sampler_obs=sa_obs, n_samples_obs=500
)
tevo.solver("rk45", (0.0, 2.0), dt=0.01, adaptive=False)

obs = {"Sx": obs_sx, "Sy": obs_sy, "Sz": obs_sz}

tevo.run(2.0, out="test", obs=obs, step_size=1e-10)
