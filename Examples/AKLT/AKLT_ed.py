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

# Tutorial on the AKLT model
#
# Using Exact Diagonalization in this module to obtain lowest three eigenvalues.
#
# References:
# - Ian Affleck, Tom Kennedy, Elliott H. Lieb, Hal Tasaki
#   Rigorous results on valence-bond ground states in antiferromagnets
#   Phys. Rev. Lett. 59, 799 (1987)
# - Ian Affleck, Tom Kennedy, Elliott H. Lieb, Hal Tasaki
#   Valence bond ground states in isotropic quantum antiferromagnets
#   Commun. Math. Phys. 115, 477 (1988)

import numpy as np
import netket as nk

# Exact ground state energy of AKLT model is zero by construction, see above references.

Sz = [[1, 0, 0], [0, 0, 0], [0, 0, -1]]
Sup = [[0, np.sqrt(2), 0], [0, 0, np.sqrt(2)], [0, 0, 0]]
Sdn = [[0, 0, 0], [np.sqrt(2), 0, 0], [0, np.sqrt(2), 0]]

# Heisenberg term
heisenberg = 0.5 * (np.kron(Sup, Sdn) + np.kron(Sdn, Sup)) + np.kron(Sz, Sz)
# AKLT two-site projector
P2_AKLT = 0.5 * heisenberg + np.dot(heisenberg, heisenberg) / 6.0 + np.identity(9) / 3.0

# 1D Lattice
g = nk.graph.Hypercube(length=10, n_dim=1, pbc=True)

# Hilbert space of spin-1s on the graph
hi = nk.hilbert.Spin(s=1, graph=g)

# AKLT model Hamiltonian as graph
ha = nk.operator.GraphOperator(hilbert=hi, bondops=[P2_AKLT.tolist()])

# Perform Lanczos Exact Diagonalization to get lowest three eigenvalues
res = nk.exact.lanczos_ed(ha, first_n=3, compute_eigenvectors=True)

# Print eigenvalues
print("eigenvalues:", res.eigenvalues)

# Compute energy of ground state
print("ground state energy:", res.mean(ha, 0))

# Compute energy of first excited state
print("first excited energy:", res.mean(ha, 1))
