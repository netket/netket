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

import pytest
import numpy as np

import netket as nk

from .. import common

pytestmark = common.skipif_mpi

SEED = 3141592
L = 4

sx = [[0, 1], [1, 0]]
sy = [[0, -1j], [1j, 0]]
sz = [[1, 0], [0, -1]]

sigmam = [[0, 0], [1, 0]]


@pytest.fixture
def liouvillian():
    hi = nk.hilbert.Spin(s=0.5) ** L

    ha = nk.operator.LocalOperator(hi)
    j_ops = []
    for i in range(L):
        ha += (0.3 / 2.0) * nk.operator.LocalOperator(hi, sx, [i])
        ha += (2.0 / 4.0) * nk.operator.LocalOperator(
            hi, np.kron(sz, sz), [i, (i + 1) % L]
        )
        j_ops.append(nk.operator.LocalOperator(hi, sigmam, [i]))

    # Create the liouvillian
    lind = nk.operator.LocalLiouvillian(ha, j_ops)
    return lind


@pytest.mark.parametrize("sparse", [True, False])
def test_exact_ss_ed(liouvillian, sparse):
    lind = liouvillian

    dm_ss = nk.exact.steady_state(lind, method="ed", sparse=sparse)
    Lop = lind.to_sparse()

    mat = np.abs(Lop @ dm_ss.reshape(-1))
    np.testing.assert_allclose(dm_ss.trace() - 1, 0.0, rtol=1e-5, atol=1e-8)

    # it does not work for sparse
    if not sparse:
        with pytest.warns(UserWarning):
            nk.exact.steady_state(lind, method="ed", sparse=sparse)
    else:
        np.testing.assert_allclose(mat, 0.0, rtol=1e-4, atol=1e-4)

    # dm_ss_d = nk.exact.steady_state(lind, method="ed", sparse=False)
    # Lop = lind.to_sparse()

    # mat = np.abs(dm_ss - dm_ss_d)
    # print(mat)
    # np.testing.assert_allclose(mat, 0.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("rho0", [False, True])
def test_exact_ss_iterative(liouvillian, sparse, rho0):
    lind = liouvillian
    M = liouvillian.hilbert.physical.n_states

    if rho0:
        rho0 = np.zeros((M, M), dtype=liouvillian.dtype)
        rho0[1:2] = 0.4
    else:
        rho0 = None

    dm_ss = nk.exact.steady_state(
        lind, sparse=sparse, method="iterative", atol=1e-5, tol=1e-5, rho0=rho0
    )
    Lop = lind.to_linear_operator()

    mat = np.abs(Lop @ dm_ss.reshape(-1))
    np.testing.assert_allclose(mat, 0.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dm_ss.trace() - 1, 0.0, rtol=1e-5, atol=1e-5)
