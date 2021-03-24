# Copyright 2021 The NetKet Authors - All rights reserved.

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

# 1D Lattice
L = 64

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=0.0)

# RBM Spin Machine
ma = nk.models.Jastrow(dtype=np.float64)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.05)
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational monte carlo driver
gs = nk.VMC(ha, op, sa, ma, n_samples=8000, sr=sr)

# Run the optimization for 300 iterations
gs.run(
    n_iter=300,
    out="test",
    # stop if variance is essentially zero (= reached eigenstate)
    callback=nk.callbacks.EarlyStopping(
        monitor="variance", baseline=1e-12, patience=np.infty
    ),
)
