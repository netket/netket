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
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg
from numpy import linalg as lalg
import pytest
from pytest import approx
import os

test_jax = True

np.set_printoptions(linewidth=180)
rg = nk.utils.RandomEngine(seed=1234)

# 1D Lattice
L = 5
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)

# Hilbert space of spins on the graph
hi = nk.hilbert.PySpin(s=0.5, graph=g)
hi_c = nk.hilbert.Spin(s=0.5, graph=g)

# Defining the Ising hamiltonian (with sign problem here)
# Using local operators
sx = [[0, 1], [1, 0]]
sy = [[0, -1j], [1j, 0]]
sz = [[1, 0], [0, -1]]

sigmam = [[0, 0], [1, 0]]

ha = nk.operator.LocalOperator(hi)
j_ops = []

for i in range(L):
    ha += nk.operator.LocalOperator(hi, sx, [i])
    ha += nk.operator.LocalOperator(hi, np.kron(sz, sz), [i, (i + 1) % L])
    j_ops.append(nk.operator.LocalOperator(hi, sigmam, [i]))


#  Create the lindbladian with
lind = nk.operator.LocalLiouvillian(ha, j_ops)


def test_der_log_val_batched_jax():
    if not test_jax:
        return

    ma = nk.machine.density_matrix.JaxNdmSpin(hilbert=hi, alpha=1, beta=1)
    ma.init_random_parameters(seed=1234, sigma=0.01)

    # test single input
    states = np.empty((5, hi.size * 2), dtype=np.float64)
    for i in range(0, 5):
        states[i, :] = lind.hilbert.number_to_state(i)

    der_loc_notc_vals = nk.operator.der_local_values(
        lind, ma, states, center_derivative=False
    )

    der_loc_vals = nk.operator.der_local_values(
        lind, ma, states, center_derivative=True
    )

    for i in range(0, states.shape[0]):
        print("doing ", i)
        state = np.atleast_2d(states[i, :])

        grad_all = nk.operator.der_local_values(
            lind, ma, state, center_derivative=False
        )

        nk._trees2_map(
            lambda x, y: np.testing.assert_array_almost_equal(
                x.flatten(), y[i].flatten()
            ),
            grad_all,
            der_loc_notc_vals,
        )

        grad_all = nk.operator.der_local_values(lind, ma, state, center_derivative=True)

        nk._trees2_map(
            lambda x, y: np.testing.assert_array_almost_equal(
                x.flatten(), y[i].flatten()
            ),
            grad_all,
            der_loc_vals,
        )
