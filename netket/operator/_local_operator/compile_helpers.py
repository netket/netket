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
from scipy import sparse

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

    op_n_conns_offdiag = max_nonzero_per_row(operators, mel_cutoff)

    # Support empty LocalOperators such as the identity.
    if len(acting_size) > 0:
        max_acting_on_sz = np.max(acting_size)
        max_local_hilbert_size = max(
            [max(map(hilbert.size_at_index, aon)) for aon in op_acting_on]
        )
        max_op_size = max(map(lambda x: x.shape[0], operators))
        max_op_size_offdiag = np.max(op_n_conns_offdiag)
    else:
        max_acting_on_sz = 0
        max_local_hilbert_size = 0
        max_op_size = 0
        max_op_size_offdiag = 0

    acting_on = np.full((n_operators, max_acting_on_sz), -1, dtype=np.intp)
    for i, aon in enumerate(op_acting_on):
        acting_on[i][: len(aon)] = aon

    local_states = np.full(
        (n_operators, max_acting_on_sz, max_local_hilbert_size), np.nan
    )
    basis = np.full((n_operators, max_acting_on_sz), 0x7FFFFFFF, dtype=np.int64)

    diag_mels = np.full((n_operators, max_op_size), np.nan, dtype=dtype)

    mels = np.full(
        (n_operators, max_op_size, max_op_size_offdiag),
        np.nan,
        dtype=dtype,
    )
    x_prime = np.full(
        (n_operators, max_op_size, max_op_size_offdiag, max_acting_on_sz),
        -1,
        dtype=np.float64,
    )
    n_conns = np.full((n_operators, max_op_size), 0, dtype=np.intp)

    for i, (aon, op) in enumerate(operators_dict.items()):
        aon_size = len(aon)
        n_local_states_per_site = np.asarray([hilbert.size_at_index(i) for i in aon])

        ## add an operator to local_states
        for j, site in enumerate(aon):
            local_states[i, j, : hilbert.shape[site]] = np.asarray(
                hilbert.states_at_index(site)
            )

        ba = 1
        for s in range(aon_size):
            basis[i, s] = ba
            ba *= hilbert.shape[aon[aon_size - s - 1]]

        if sparse.issparse(op):
            if not isinstance(op, sparse.csr_matrix):
                op = op.tocsr()
            # Extract the sparse matrix representation to numpy arrays
            data = np.array(op.data, copy=False)
            indices = np.array(op.indices, copy=False)
            indptr = np.array(op.indptr, copy=False)

            _append_matrix_sparse(
                data,
                indices,
                indptr,
                aon_size,
                local_states[i],
                n_local_states_per_site,
                mel_cutoff,
                diag_mels[i],
                mels[i],
                x_prime[i],
                n_conns[i],
            )

        else:
            _append_matrix(
                op,
                aon_size,
                local_states[i],
                n_local_states_per_site,
                mel_cutoff,
                diag_mels[i],
                mels[i],
                x_prime[i],
                n_conns[i],
            )

    nonzero_diagonal = (
        np.any(np.abs(diag_mels) >= mel_cutoff) or np.abs(constant) >= mel_cutoff
    )

    max_conn_size = 1 if nonzero_diagonal else 0
    max_conn_size = max_conn_size + np.sum(op_n_conns_offdiag)

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
    acting_size,
    local_states_per_site,
    hilb_size_per_site,
    epsilon,
    diag_mels,
    mels,
    x_prime,
    n_conns,
):
    """
    Appends
    """
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
def _append_matrix_sparse(
    data,
    indices,
    indptr,
    acting_size,
    local_states_per_site,
    hilb_size_per_site,
    epsilon,
    diag_mels,
    mels,
    x_prime,
    n_conns,
):
    """
    Equivalent to _append_matrix, but takes as input the three arrays
    'data, indices, indptr' of the CSR sparse format instead of a numpy
    dense matrix.
    """
    op_size = len(indptr) - 1

    for i in range(op_size):
        # If the diagonal element was not found in the data, set it to 0
        diag_mels[i] = 0
        for index in range(indptr[i], indptr[i + 1]):
            j = indices[index]
            val = data[index]

            if i == j:  # Diagonal elements
                diag_mels[i] = val

            elif np.abs(val) > epsilon:  # Non-diagonal elements
                k_conn = n_conns[i]
                mels[i, k_conn] = val
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


# Generated by chatting with chatgpt
# https://chat.openai.com/share/9fb4b0d3-0306-41ca-aefc-eee780a2dc02
def max_nonzero_per_row(operators, cutoff):
    """
    This function counts the maximum number of nonzero entries per row, excluding the
    diagonal, among all operators in a local operator.

    Prior to this function, netket defaulted to returning the maximum shape of the
    operators minus 1, but that was largely memory inefficient.

    This function has a computational overhead, as we have to iterate through all
    operators, but it allows us to be much happier in terms of memory cost.
    """
    max_counts = []
    for matrix in operators:
        # Check if the matrix is sparse
        if sparse.issparse(matrix):
            # simple implementation, raises warning
            # matrix = matrix.copy()  # Ensure we don't modify the original matrix
            # matrix.data[np.abs(matrix.data) < cutoff] = 0  # Apply cutoff
            # matrix.setdiag(0)  # Set diagonal entries to zero

            # alternative implementation. Does not raise warning but
            # I'm unsure if it's any faster...
            # Convert to dok_matrix format for efficient modification
            dok_matrix = matrix.todok()
            dok_matrix.setdiag(0)

            # Collect keys to delete
            keys_to_delete = [
                (i, j) for (i, j), v in dok_matrix.items() if abs(v) < cutoff
            ]

            # Delete keys
            for key in keys_to_delete:
                del dok_matrix[key]

            row_counts = dok_matrix.tocsr().getnnz(
                axis=1
            )  # Count non-zero entries in each row
        else:
            matrix = matrix.copy()  # Make a copy to avoid modifying the original matrix
            mask = np.abs(matrix) >= cutoff  # Create a mask of entries above the cutoff
            np.fill_diagonal(mask, 0)  # Set diagonal entries to zero
            row_counts = np.count_nonzero(
                mask, axis=1
            )  # Count non-zero entries in each row

        max_counts.append(np.max(row_counts))
    return np.array(max_counts, dtype=np.int32)
