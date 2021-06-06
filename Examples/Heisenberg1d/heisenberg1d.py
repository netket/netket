# Copyright 2021 The NetKet Authors - All rights reserved.
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

import netket as nk
import numpy as np
import jax

# 1D Lattice
L = 20

g = nk.graph.Chain(length=L, pbc=True)
# Lattice translation operations

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Ising spin hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

# RBM Spin Machine
ma = nk.models.RBMSymm(
    symmetries=g.translations(),
    alpha=4,
    use_visible_bias=False,
    use_hidden_bias=True,
    dtype=float,
)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisExchange(hi, graph=g, n_chains=16)

# Optimizer
op = nk.optim.Sgd(learning_rate=0.01)
sr = nk.optim.SR(diag_shift=0.1)

# Variational monte carlo driver
gs = nk.VMC(ha, op, sa, ma, n_samples=1000, n_discard_per_chain=100, preconditioner=sr)

# Print parameter structure
print(f"# variational parameters: {gs.state.n_parameters}")


# Run the optimization for 300 iterations
gs.run(n_iter=300, out="test")
