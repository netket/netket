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
import jax.numpy as jnp

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

    # how many sites each operator is acting on
    acting_size = np.array([len(aon) for aon in op_acting_on], dtype=np.intp)

    # compute the maximum number of off-diagonal nonzeros (over all rows) of each operator
    op_n_conns_offdiag = max_nonzero_per_row(operators, mel_cutoff)

    # Support empty LocalOperators such as the identity.
    if len(acting_size) > 0:
        # maximum number of sites any operator is acting on
        max_acting_on_sz = np.max(acting_size)

        # maximum size of any operator (maximum size of the matrix / prod of local hilbert spaces)
        max_op_size = max(map(lambda x: x.shape[0], operators))
        # maximum number of off-diagonal nonzeros of any operator
        max_op_size_offdiag = np.max(op_n_conns_offdiag)
    else:
        max_acting_on_sz = 0
        max_op_size = 0
        max_op_size_offdiag = 0

    # matrix storing which sites each operator acts on, padded with -1
    acting_on = np.full((n_operators, max_acting_on_sz), -1, dtype=np.intp)
    for i, aon in enumerate(op_acting_on):
        acting_on[i][: len(aon)] = aon

    ###
    # allocate empty arrays which are filled below

    # array storing the basis for each site each operator is acting on
    # The basis is an integer used to map (indices of) local states on sites
    # to (indices of) states in the space spanned by all sites the op is acting on
    # (it is simply the product of number of local states of the sites before)
    basis = np.full((n_operators, max_acting_on_sz), 0x7FFFFFFF, dtype=np.int64)

    diag_mels = np.full((n_operators, max_op_size), np.nan, dtype=dtype)

    mels = np.full(
        (n_operators, max_op_size, max_op_size_offdiag),
        np.nan,
        dtype=dtype,
    )
    # x_prime contains the local state after the operator has been applied
    # for the sites the operator is acting on, for each row
    # (each row also corresponds to a list of local states)
    x_prime = np.full(
        (n_operators, max_op_size, max_op_size_offdiag, max_acting_on_sz),
        -1,
        dtype=np.float64,
    )
    # store the number of off-diagonal nonzeros per row of each operator
    n_conns = np.full((n_operators, max_op_size), 0, dtype=np.intp)

    ###
    # iterate over all operators
    for i, (aon, op) in enumerate(operators_dict.items()):
        # how many sites this operator is acting on
        aon_size = len(aon)

        n_local_states_per_site = np.asarray([hilbert.size_at_index(i) for i in aon])

        # compute the basis of each site of this operator
        # i.e. the product of the number of local states of all sites before it
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
    # estimate max_conn_size with the sum of the
    # maximum number of off-diagonal nonzeros of all operators
    max_conn_size = max_conn_size + np.sum(op_n_conns_offdiag)

    return {
        "acting_on": acting_on,
        "acting_size": acting_size,
        "diag_mels": diag_mels,
        "mels": mels,
        "x_prime": x_prime,
        "n_conns": n_conns,
        "basis": basis,
        "nonzero_diagonal": nonzero_diagonal,
        "max_conn_size": max_conn_size,
    }


def pack_internals_jax(
    hilbert: AbstractHilbert,
    operators_dict: dict,
    constant,
    dtype: DType,
    mel_cutoff: float,
):
    # we groups together operators which act on the same number of sites
    # and then call pack_internals on each group
    #
    # TODO in the future consider separating also operators with different sizes
    # to avoid excessive padding
    # (only relevant for non-uniform number of local states)

    op_acting_on = list(operators_dict.keys())
    operators = list(operators_dict.values())
    # how many sites each operator is acting on
    acting_size = np.array([len(aon) for aon in op_acting_on], dtype=np.intp)

    data = {}
    data["nonzero_diagonal"] = np.abs(constant) >= mel_cutoff
    data["max_conn_size"] = 0

    # iterate over groups of operators with same number of sites
    # special case for empty operator
    for s in np.unique(acting_size) if len(acting_size) > 0 else [0]:
        (indices,) = np.where(acting_size == s)
        operators_dict_s = {op_acting_on[i]: operators[i] for i in indices}
        data_s = pack_internals(hilbert, operators_dict_s, 0, dtype, mel_cutoff)
        nonzero_diagonal = bool(data_s.pop("nonzero_diagonal"))
        max_conn_size = int(data_s.pop("max_conn_size"))
        if nonzero_diagonal:
            # we only count the diag elem once below, so we subtract it here
            max_conn_size = max_conn_size - 1
        data["nonzero_diagonal"] = data["nonzero_diagonal"] or nonzero_diagonal
        data["max_conn_size"] = data["max_conn_size"] + max_conn_size
        # append other elements to lists:
        for k, v in data_s.items():
            data[k] = data.pop(k, []) + [jnp.asarray(v)]

    # count the diagonal once at the end
    if data["nonzero_diagonal"]:
        data["max_conn_size"] = data["max_conn_size"] + 1

    return data


@numba.jit(nopython=True)
def _append_matrix(
    operator,
    acting_size,
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
    # iterate over rows
    for i in range(op_size):
        # set diag mels
        diag_mels[i] = operator[i, i]
        # count number of connected elements
        n_conns[i] = 0  # k_conn = 0
        # iterate over cols
        for j in range(op_size):
            # off-diagonal, non-zero
            if i != j and np.abs(operator[i, j]) > epsilon:
                k_conn = n_conns[i]
                # set off-doagnoal mels
                mels[i, k_conn] = operator[i, j]
                # convert the col to the corresponding local states
                # and store it in x_prime
                _number_to_state(
                    j,
                    hilb_size_per_site,
                    x_prime[i, k_conn, :acting_size],
                )
                n_conns[i] += 1  # k_conn=k_conn+1


@numba.jit(nopython=True)
def _append_matrix_sparse(
    data,
    indices,
    indptr,
    acting_size,
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
                    x_prime[i, k_conn, :acting_size],
                )
                n_conns[i] += 1


@numba.jit(nopython=True)
def _number_to_state(number, hilbert_size_per_site, out):
    out[:] = 0
    size = out.shape[0]

    ip = number
    k = size - 1
    while ip > 0:
        local_size = hilbert_size_per_site[k]
        out[k] = ip % local_size
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
