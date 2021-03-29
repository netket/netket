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
from scipy.sparse.linalg import eigsh

# 1D Lattice
g = nk.graph.Hypercube(length=16, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Heisenberg spin hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)

# Convert hamiltonian to a sparse matrix
# Here we further take only the real part since the Heisenberg Hamiltonian is real
sp_ha = ha.to_sparse().real

# Use scipy sparse diagonalization
vals, vecs = eigsh(sp_ha, k=2, which="SA")
print("eigenvalues with scipy sparse:", vals)

# Explicitely compute energy of ground state
# Doing full dot product
psi = vecs[:, 0]
print("\ng.s. energy:", psi @ sp_ha @ psi)

# Compute energy of first excited state
psi = vecs[:, 1]
print("\nfirst excited energy:", psi @ sp_ha @ psi)
