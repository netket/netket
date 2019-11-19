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


np.set_printoptions(linewidth=180)
rg = nk.utils.RandomEngine(seed=1234)

# 1D Lattice
L = 5
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5, graph=g)


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


# Create the lindbladian with no jump operators
lind = nk.operator.LocalLindbladian(ha)

# add the jump operators
for j_op in j_ops:
    lind.add_jump_op(j_op)


def test_lindblad_form():
    ## Construct the lindbladian by hand:
    idmat = sparse.eye(2**L)

    # Build the non hermitian matrix
    hnh_mat = ha.to_sparse()
    for j_op in j_ops:
        j_mat = j_op.to_sparse()
        hnh_mat -= 0.5j * j_mat.H*j_mat


    # Compute the left and right product with identity
    lind_mat = -1j*sparse.kron(idmat, hnh_mat) + 1j*sparse.kron(hnh_mat.H, idmat) 
    # add jump operators
    for j_op in j_ops:
        j_mat = j_op.to_sparse()
        lind_mat += sparse.kron(j_mat.conj(), j_mat)


    assert (lind_mat.todense() == lind.to_dense()).all()


def test_lindblad_zero_eigenvalue():
    lind_mat = lind.to_sparse()
    w, v = linalg.eigsh(lind_mat.H*lind_mat, which='SM')
    assert w[0] <= 10e-10


def test_der_log_val():
    ma = nk.machine.NdmSpinPhase(hilbert=hi, alpha=1, beta=1)
    ma.init_random_parameters(seed=1234, sigma=0.01)

    for i in range(0,lind.hilbert.n_states):
        state = lind.hilbert.number_to_state(i)
        der_loc_vals = nk.operator.der_local_values(lind, ma, state)

        log_val_s = ma.log_val(state)
        der_log_s = ma.der_log(state)

        delta, mel = lind.get_conn(state)
        statet = state + delta

        log_val_p = ma.log_val(statet)
        der_log_p = ma.der_log(statet)

        log_val_diff = mel * np.exp(log_val_p - log_val_s)
        log_val_diff = log_val_diff.reshape((log_val_diff.size, 1))

        grad = log_val_diff * (der_log_p - der_log_s)
        grad_all = grad.sum(axis=0)
        assert (grad_all == der_loc_vals).all()