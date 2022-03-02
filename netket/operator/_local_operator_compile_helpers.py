# Copyright 2022 The NetKet Authors - All rights reserved.
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

"""
This file contains functions generating the numba-packed representation of local
operators.
"""

import numpy as np
import numba

from netket.hilbert import AbstractHilbert
from netket.utils.types import DType


def pack_internals(
    hilbert: AbstractHilbert,
    operators_dict: dict,
    constant,
    dtype: DType,
    mel_cutoff: float,
):
    """
    Take the internal lazy representation of a local operator and returns the arrays
    needed for the numba implementation.

    This takes as input a dictionary with Tuples as keys, the `acting_on` and matrices as values.
    The keys represent the sites upon which the matrix acts.
    It is assumed that the integer in the tuples are sorted.

    Returns a dictionary with all the data fields
    """
    op_acting_on = list(operators_dict.keys())
    operators = list(operators_dict.values())
    n_operators = len(operators_dict)

    """Analyze the operator strings and precompute arrays for get_conn inference"""
    acting_size = np.array([len(aon) for aon in op_acting_on], dtype=np.intp)
    max_acting_on_sz = np.max(acting_size)
    max_local_hilbert_size = max(
        [max(map(hilbert.size_at_index, aon)) for aon in op_acting_on]
    )
    max_op_size = max(map(lambda x: x.shape[0], operators))

    acting_on = np.full((n_operators, max_acting_on_sz), -1, dtype=np.intp)
    for (i, aon) in enumerate(op_acting_on):
        acting_on[i][: len(aon)] = aon

    local_states = np.full(
        (n_operators, max_acting_on_sz, max_local_hilbert_size), np.nan
    )
    basis = np.full((n_operators, max_acting_on_sz), 1e10, dtype=np.int64)

    diag_mels = np.full((n_operators, max_op_size), np.nan, dtype=dtype)
    mels = np.full(
        (n_operators, max_op_size, max_op_size - 1),
        np.nan,
        dtype=dtype,
    )
    x_prime = np.full(
        (n_operators, max_op_size, max_op_size - 1, max_acting_on_sz),
        -1,
        dtype=np.float64,
    )
    n_conns = np.full((n_operators, max_op_size), -1, dtype=np.intp)

    for (i, (aon, op)) in enumerate(operators_dict.items()):
        aon_size = len(aon)
        n_local_states_per_site = np.asarray([hilbert.size_at_index(i) for i in aon])

        ## add an operator to local_states
        for (j, site) in enumerate(aon):
            local_states[i, j, : hilbert.shape[site]] = np.asarray(
                hilbert.states_at_index(site)
            )

        ba = 1
        for s in range(aon_size):
            basis[i, s] = ba
            ba *= hilbert.shape[aon_size - s - 1]

        # eventually could support sparse matrices
        # if isinstance(op, sparse.spmatrix):
        #    op = op.todense()

        _append_matrix(
            op,
            diag_mels[i],
            mels[i],
            x_prime[i],
            n_conns[i],
            aon_size,
            local_states[i],
            mel_cutoff,
            n_local_states_per_site,
        )

    nonzero_diagonal = (
        np.any(np.abs(diag_mels) >= mel_cutoff) or np.abs(constant) >= mel_cutoff
    )

    max_conn_size = 1 if nonzero_diagonal else 0
    for op in operators:
        nnz_mat = np.abs(op) > mel_cutoff
        nnz_mat[np.diag_indices(nnz_mat.shape[0])] = False
        nnz_rows = np.sum(nnz_mat, axis=1)
        max_conn_size += np.max(nnz_rows)

    return {
        "acting_on": acting_on,
        "acting_size": acting_size,
        "diag_mels": diag_mels,
        "mels": mels,
        "x_prime": x_prime,
        "n_conns": n_conns,
        "local_states": local_states,
        "basis": basis,
        "nonzero_diagonal": nonzero_diagonal,
        "max_conn_size": max_conn_size,
    }


@numba.jit(nopython=True)
def _append_matrix(
    operator,
    diag_mels,
    mels,
    x_prime,
    n_conns,
    acting_size,
    local_states_per_site,
    epsilon,
    hilb_size_per_site,
):
    op_size = operator.shape[0]
    assert op_size == operator.shape[1]
    for i in range(op_size):
        diag_mels[i] = operator[i, i]
        n_conns[i] = 0
        for j in range(op_size):
            if i != j and np.abs(operator[i, j]) > epsilon:
                k_conn = n_conns[i]
                mels[i, k_conn] = operator[i, j]
                _number_to_state(
                    j,
                    hilb_size_per_site,
                    local_states_per_site[:acting_size, :],
                    x_prime[i, k_conn, :acting_size],
                )
                n_conns[i] += 1


@numba.jit(nopython=True)
def _number_to_state(number, hilbert_size_per_site, local_states_per_site, out):

    out[:] = local_states_per_site[:, 0]
    size = out.shape[0]

    ip = number
    k = size - 1
    while ip > 0:
        local_size = hilbert_size_per_site[k]
        out[k] = local_states_per_site[k, ip % local_size]
        ip = ip // local_size
        k -= 1

    return out
