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

# 1D Lattice
L = 12  # 10

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0).to_local_operator()


# the pauli matrices are hard coded for simplicity
sx = np.array([[0, 1], [1, 0]])
sz = np.array([[1, 0], [0, -1]])
sy = np.array([[0, -1j], [1j, 0]])
zz = np.kron(sz, sz)
yy = np.kron(sy, sy)
xx = np.kron(sx, sx)
jj = np.real(xx + yy + zz)


def create_Hamiltonain_operator_list():
    operators = []
    sites = []

    global sx, sz, jj

    mu = 0.08333333333333333
    clatt = {
        (0.0, 4.0): -1.0,
        (0.0, 5.0): -1.0,
        (0.0, 6.0): -1.0,
        (0.0, 7.0): -1.0,
        (0.0, 8.0): 0.0,
        (0.0, 9.0): 0.0,
        (0.0, 10.0): 0.0,
        (0.0, 11.0): 0.0,
        (1.0, 4.0): -1.0,
        (1.0, 5.0): -1.0,
        (1.0, 6.0): -1.0,
        (1.0, 7.0): -1.0,
        (1.0, 8.0): 0.0,
        (1.0, 9.0): 0.0,
        (1.0, 10.0): 0.0,
        (1.0, 11.0): 0.0,
        (2.0, 4.0): -1.0,
        (2.0, 5.0): -1.0,
        (2.0, 6.0): -1.0,
        (2.0, 7.0): -1.0,
        (2.0, 8.0): 0.0,
        (2.0, 9.0): 0.0,
        (2.0, 10.0): 0.0,
        (2.0, 11.0): 0.0,
        (3.0, 4.0): -1.0,
        (3.0, 5.0): -1.0,
        (3.0, 6.0): -1.0,
        (3.0, 7.0): -1.0,
        (3.0, 8.0): 0.0,
        (3.0, 9.0): 0.0,
        (3.0, 10.0): 0.0,
        (3.0, 11.0): 0.0,
        (4.0, 8.0): 0.0,
        (4.0, 9.0): 0.0,
        (4.0, 10.0): 0.0,
        (4.0, 11.0): 0.0,
        (5.0, 8.0): 0.0,
        (5.0, 9.0): 0.0,
        (5.0, 10.0): 0.0,
        (5.0, 11.0): 0.0,
        (6.0, 8.0): 0.0,
        (6.0, 9.0): 0.0,
        (6.0, 10.0): 0.0,
        (6.0, 11.0): 0.0,
        (7.0, 8.0): 0.0,
        (7.0, 9.0): 0.0,
        (7.0, 10.0): 0.0,
        (7.0, 11.0): 0.0,
    }

    # two-body interaction
    for key in clatt:
        operators.append((mu * (1 - clatt[key[0], key[1]]) * jj).tolist())
        sites.append([key[0], key[1]])
    return sites, operators


def build_Hamiltonian(hilbert):
    sites, operators = create_Hamiltonain_operator_list()
    Hamiltonian = nk.operator.LocalOperator(hilbert, operators, sites)
    return Hamiltonian


ha = build_Hamiltonian(hi)

# RBM Spin Machine
ma = nk.models.RBM(alpha=1, use_visible_bias=True, dtype=float)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.1)

# Variational monte carlo driver
gs = nk.VMC(ha, op, sa, ma, n_samples=2, n_discard=0)

# Run the optimization for 300 iterations
gs.run(n_iter=300, out="test")
