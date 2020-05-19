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
L = 3
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


def test_der_log_val():
    ma = nk.machine.density_matrix.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(seed=1234, sigma=0.01)

    # test single input
    for i in range(0, lind.hilbert.n_states):
        state = np.atleast_2d(lind.hilbert.number_to_state(i))
        der_loc_vals = nk.operator.der_local_values(
            lind, ma, state, center_derivative=False
        )

        log_val_s = ma.log_val(state)[0]
        der_log_s = ma.der_log(state)[0]

        statet, mel = lind.get_conn(state[0, :])

        log_val_p = ma.log_val(statet)
        der_log_p = ma.der_log(statet)

        log_val_diff = mel * np.exp(log_val_p - log_val_s)
        log_val_diff = log_val_diff.reshape((log_val_diff.size, 1))

        grad = log_val_diff * (
            der_log_p
        )  # - der_log_s) because derivative not centered
        grad_all = grad.sum(axis=0)

        np.testing.assert_array_almost_equal(grad_all, der_loc_vals.flatten())

        # centered
        # not necessary for liouvillian but worth checking
        der_loc_vals = nk.operator.der_local_values(
            lind, ma, state, center_derivative=True
        )
        grad = log_val_diff * (der_log_p - der_log_s)
        grad_all = grad.sum(axis=0)

        np.testing.assert_array_almost_equal(grad_all, der_loc_vals.flatten())


def test_der_log_val_batched():
    ma = nk.machine.density_matrix.RbmSpin(hilbert=hi, alpha=1)
    ma.init_random_parameters(seed=1234, sigma=0.01)

    states = np.empty((5, hi_c.size * 2), dtype=np.float64)
    der_locs = np.empty((5, ma.n_par), dtype=np.complex128)
    der_locs_c = np.empty((5, ma.n_par), dtype=np.complex128)
    # test single input
    for i in range(0, 5):
        state = lind.hilbert.number_to_state(i)
        states[i, :] = state
        der_locs[i, :] = nk.operator.der_local_values(
            lind, ma, np.atleast_2d(state), center_derivative=False
        )
        der_locs_c[i, :] = nk.operator.der_local_values(
            lind, ma, np.atleast_2d(state), center_derivative=True
        )

    der_locs_all = nk.operator.der_local_values(
        lind, ma, states, center_derivative=False
    )
    der_locs_all_c = nk.operator.der_local_values(
        lind, ma, states, center_derivative=True
    )

    np.testing.assert_array_almost_equal(der_locs, der_locs_all)
    np.testing.assert_array_almost_equal(der_locs_c, der_locs_all_c)


if test_jax:
    import jax
    import jax.experimental
    import jax.experimental.stax


def test_der_log_val_jax():
    if not test_jax:
        return

    ma = nk.machine.density_matrix.JaxNdmSpin(hilbert=hi, alpha=1, beta=1)
    ma.init_random_parameters(seed=1234, sigma=0.01)

    # test single input
    for i in range(0, lind.hilbert.n_states):
        state = jax.numpy.array(np.atleast_2d(lind.hilbert.number_to_state(i)))
        der_loc_vals = nk.operator.der_local_values(
            lind, ma, state, center_derivative=False
        )

        log_val_s = ma.log_val(state)
        der_log_s = ma.der_log(state)

        statet, mel = lind.get_conn(state[0, :]._value)

        log_val_p = ma.log_val(statet)
        der_log_p = ma.der_log(statet)

        log_val_diff = mel * np.exp(log_val_p - log_val_s)
        log_val_diff = log_val_diff.reshape((log_val_diff.size, 1))

        grad_all = nk._tree_map(
            lambda x: (
                log_val_diff.reshape((-1,) + tuple(1 for i in range(x.ndim - 1))) * x
            ).sum(axis=0),
            der_log_p,
        )

        nk._trees2_map(
            lambda x, y: np.testing.assert_array_almost_equal(x.flatten(), y.flatten()),
            grad_all,
            der_loc_vals,
        )

        # centered
        # not necessary for liouvillian but worth checking
        der_loc_vals = nk.operator.der_local_values(
            lind, ma, state, center_derivative=True
        )
        grad_all = nk._trees2_map(
            lambda xp, x: (
                log_val_diff.reshape((-1,) + tuple(1 for i in range(x.ndim - 1)))
                * (xp - x)
            ).sum(axis=0),
            der_log_p,
            der_log_s,
        )

        nk._trees2_map(
            lambda x, y: np.testing.assert_array_almost_equal(x.flatten(), y.flatten()),
            grad_all,
            der_loc_vals,
        )


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
        lind, ma, jax.numpy.array(states), center_derivative=False
    )

    der_loc_vals = nk.operator.der_local_values(
        lind, ma, jax.numpy.array(states), center_derivative=True
    )

    for i in range(0, states.shape[0]):
        print("doing ", i)
        state = jax.numpy.array(np.atleast_2d(states[i, :]))

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
