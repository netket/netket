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

import netket.legacy as nk
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg
from numpy import linalg as lalg
import pytest
from pytest import approx
import os

np.set_printoptions(linewidth=180)

# 1D Lattice
L = 4

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5) ** L

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


def test_lindblad_form():
    ## Construct the lindbladian by hand:
    idmat = sparse.eye(2 ** L)

    # Build the non hermitian matrix
    hnh_mat = ha.to_sparse()
    for j_op in j_ops:
        j_mat = j_op.to_sparse()
        hnh_mat -= 0.5j * j_mat.H * j_mat

    # Compute the left and right product with identity
    lind_mat = -1j * sparse.kron(idmat, hnh_mat) + 1j * sparse.kron(hnh_mat.H, idmat)
    # add jump operators
    for j_op in j_ops:
        j_mat = j_op.to_sparse()
        lind_mat += sparse.kron(j_mat.conj(), j_mat)

    assert (lind_mat.todense() == lind.to_dense()).all()


def test_lindblad_zero_eigenvalue():
    lind_mat = lind.to_sparse()
    w, v = linalg.eigsh(lind_mat.H * lind_mat, which="SM")
    assert w[0] <= 10e-10


def test_linear_operator():
    l_sparse = lind.to_dense()
    l_op = lind.to_linear_operator()

    dm = np.random.rand(hi.n_states, hi.n_states) + 1j * np.random.rand(
        hi.n_states, hi.n_states
    )
    dm = (dm + dm.T.conj()).reshape(-1)

    res_sparse = l_sparse @ dm
    res_op = l_op @ dm

    assert np.all(res_sparse - res_op == approx(0.0, rel=1e-6, abs=1e-6))

    assert res_sparse.reshape((hi.n_states, hi.n_states)).trace() == approx(
        0.0, rel=1e-6, abs=1e-6
    )
    assert res_op.reshape((hi.n_states, hi.n_states)).trace() == approx(
        0.0, rel=1e-6, abs=1e-6
    )

    l_op = lind.to_linear_operator(append_trace=True)
    dmptr = np.zeros(dm.size + 1, dtype=dm.dtype).reshape(-1)
    dmptr[:-1] = dm
    res_op2 = l_op @ dmptr

    assert np.all(res_op2[:-1] - res_op == approx(0.0, rel=1e-8, abs=1e-8))
    assert res_op2[-1] - dm.reshape((hi.n_states, hi.n_states)).trace() == approx(
        0.0, rel=1e-8, abs=1e-8
    )


# Construct the operators for Sx, Sy and Sz
obs_sx = nk.operator.LocalOperator(hi)
obs_sy = nk.operator.LocalOperator(hi)
obs_sz = nk.operator.LocalOperator(hi)
for i in range(L):
    obs_sx += nk.operator.LocalOperator(hi, sx, [i])
    obs_sy += nk.operator.LocalOperator(hi, sy, [i])
    obs_sz += nk.operator.LocalOperator(hi, sz, [i])


sxmat = obs_sx.to_dense()
symat = obs_sy.to_dense()
szmat = obs_sz.to_dense()
