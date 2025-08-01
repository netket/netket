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

import netket as nk
import numpy as np
from scipy import sparse

import pytest
from pytest import approx

from netket.operator import spin, LocalOperatorNumba

from test import common

pytestmark = common.skipif_distributed

np.set_printoptions(linewidth=180)

# 1D Lattice
L = 4

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=0.5) ** L

ha = nk.operator.LocalOperatorNumba(hi, dtype=complex)
j_ops = []

for i in range(L):
    ha += spin.sigmax(hi, i, cls=LocalOperatorNumba)
    ha += spin.sigmay(hi, i, cls=LocalOperatorNumba)
    ha += spin.sigmaz(hi, i, cls=LocalOperatorNumba) @ spin.sigmaz(
        hi, (i + 1) % L, cls=LocalOperatorNumba
    )
    j_ops.append(spin.sigmam(hi, i, cls=LocalOperatorNumba))
    j_ops.append(1j * spin.sigmam(hi, i, cls=LocalOperatorNumba))


# Create the lindbladian with
lind = nk.operator.LocalLiouvillian(ha, j_ops)


def test_lindblad_form():
    ## Construct the lindbladian by hand:
    idmat = sparse.eye(2**L)

    # Build the non hermitian matrix
    hnh_mat = ha.to_sparse()
    for j_op in j_ops:
        j_mat = j_op.to_sparse()
        hnh_mat -= 0.5j * j_mat.T.conj() * j_mat

    # Compute the left and right product with identity
    lind_mat = -1j * sparse.kron(hnh_mat, idmat) + 1j * sparse.kron(
        idmat, hnh_mat.conj()
    )
    # add jump operators
    for j_op in j_ops:
        j_mat = j_op.to_sparse()
        lind_mat += sparse.kron(j_mat, j_mat.conj())

    print(lind_mat.shape)
    print(lind.to_dense().shape)
    np.testing.assert_allclose(lind_mat.todense(), lind.to_dense())


def test_liouvillian_no_dissipators():
    lind = nk.operator.LocalLiouvillian(ha)

    ## Construct the lindbladian by hand:
    idmat = sparse.eye(2**L)
    h_mat = ha.to_sparse()

    lind_mat = -1j * sparse.kron(h_mat, idmat) + 1j * sparse.kron(idmat, h_mat.conj())

    np.testing.assert_allclose(lind.to_dense(), lind_mat.todense())


def test_lindblad_zero_eigenvalue():
    lind_mat = lind.to_dense()
    w, v = np.linalg.eigh(lind_mat.T.conj() @ lind_mat)
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

    np.testing.assert_allclose(res_sparse, res_op, rtol=1e-6, atol=1e-6)

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

    np.testing.assert_allclose(res_op2[:-1], res_op, rtol=1e-8, atol=1e-8)
    assert res_op2[-1] - dm.reshape((hi.n_states, hi.n_states)).trace() == approx(
        0.0, rel=1e-8, abs=1e-8
    )


dtypes_r = [np.float32, np.float64]
dtypes_c = [np.complex64, np.complex128]
dtypes = dtypes_r + dtypes_c


@pytest.mark.parametrize("dtype", dtypes)
def test_dtype(dtype):
    if not nk.jax.is_complex_dtype(dtype):
        with pytest.raises(TypeError):
            lind = nk.operator.LocalLiouvillian(ha, j_ops, dtype=dtype)
        return
    else:
        lind = nk.operator.LocalLiouvillian(ha, j_ops, dtype=dtype)
        dtype_c = dtype

    assert lind.dtype == dtype_c
    assert lind.hamiltonian_nh.dtype == dtype_c
    for op in lind.jump_operators:
        assert op.dtype == dtype_c

    for _dt in dtypes_r + [np.int8, np.int16]:
        sigma = op.hilbert.numbers_to_states(np.array([0, 1, 2, 3]))
        sigma = np.array(sigma, dtype=_dt)
        sigmap, mels = op.get_conn_padded(sigma)
        assert sigmap.dtype == sigma.dtype
        # TODO: fix this test
        # assert mels.dtype == lind.dtype
