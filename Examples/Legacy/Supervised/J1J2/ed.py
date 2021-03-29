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

import numpy as np
from netket import legacy as nk


def load_ed_data(L, J2=0.4):
    # Sigma^z*Sigma^z interactions
    sigmaz = np.array([[1, 0], [0, -1]])
    mszsz = np.kron(sigmaz, sigmaz)

    # Exchange interactions
    exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

    # Couplings J1 and J2
    J = [1.0, J2]

    mats = []
    sites = []

    for i in range(L):
        for d in [0, 1]:
            # \sum_i J*sigma^z(i)*sigma^z(i+d)
            mats.append((J[d] * mszsz).tolist())
            sites.append([i, (i + d + 1) % L])

            # \sum_i J*(sigma^x(i)*sigma^x(i+d) + sigma^y(i)*sigma^y(i+d))
            mats.append(((-1.0) ** (d + 1) * J[d] * exchange).tolist())
            sites.append([i, (i + d + 1) % L])

    # 1D Lattice
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

    # Spin based Hilbert Space
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

    # Custom Hamiltonian operator
    ha = nk.operator.LocalOperator(hi)
    for mat, site in zip(mats, sites):
        ha += nk.operator.LocalOperator(hi, mat, site)

    # Perform Lanczos Exact Diagonalization to get lowest three eigenvalues
    res = nk.exact.lanczos_ed(ha, first_n=3, compute_eigenvectors=True)

    # Eigenvector
    ttargets = []

    tsamples = []

    for i, state in enumerate(hi.states()):
        # only pick zero-magnetization states
        mag = np.sum(state)
        if np.abs(mag) < 1.0e-4:
            tsamples.append(state.tolist())
            ttargets.append([np.log(res.eigenvectors[0][i])])

    return hi, tsamples, ttargets
