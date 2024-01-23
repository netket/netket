# Copyright 2022-23 The NetKet Authors - All rights reserved.
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

import numbers

import numpy as np

from scipy import sparse
from scipy.sparse import spmatrix

from netket.hilbert import AbstractHilbert, Fock
from netket.jax import canonicalize_dtypes
from netket.utils.types import DType, Array


def cast_operator_matrix_dtype(matrix: Array, dtype: DType):
    """
    Changes the dtype of a matrix, without changing the structural type of the object.

    This makes sure that if you pass sparse arrays to a LocalOperator, they remain
    sparse even if you change the dtype
    """
    # must copy
    # return np.asarray(matrix, dtype=dtype)
    return matrix.astype(dtype)


def _standardize_matrix_input_type(op):
    """
    Standardize the structural type of operators stored in LocalOperator.

    Eventually, we could also support spmatrices (but some work will be needed.)
    """
    if isinstance(op, list):
        return np.asarray(op)
    elif sparse.issparse(op):
        return op.tocoo()
    elif isinstance(op, np.matrix):
        # np.matrix does not respect the ndarray interface
        return np.asarray(op)
    else:
        return op


def _kron(A, B):
    if isinstance(A, spmatrix) and isinstance(B, spmatrix):
        res = sparse.kron(A, B, format="coo")
    elif isinstance(A, spmatrix) ^ isinstance(B, spmatrix):
        # XOR, meaning only one is sparse matrix
        # Then we maintain sparse format (Is this a good idea?)
        res = sparse.kron(A, B, format="coo")
    else:
        res = np.kron(np.asarray(A), np.asarray(B))
    return res


def _eye_like(size, *, dtype, like):
    if sparse.issparse(like):
        return sparse.eye(size, dtype=dtype)
    else:
        return np.eye(size, dtype=dtype)


def canonicalize_input(
    hilbert: AbstractHilbert, operators, acting_on, constant, *, dtype=None
):
    """
    Takes as inputs the inputs to the constructor of LocalOperator and canonicalizes
    them by ensuring the following holds:
     - acting_on is a list of list
     - acting_on[i] are sorted
     - operators is list of matrices
     - all dtypes match

    Args:
        hilbert: The hilbert space

    Returns:
        List of operators, acting ons and dtypes.
    """
    # check if passing a single operator or a list of operators
    if isinstance(acting_on, numbers.Number):
        acting_on = [acting_on]

    is_nested = any(hasattr(i, "__len__") for i in acting_on)
    if not is_nested:
        operators = [operators]
        acting_on = [acting_on]

    if all(len(aon) == 0 for aon in acting_on):
        operators = []
        acting_on = []
    else:
        if max(map(max, acting_on)) >= hilbert.size or min(map(min, acting_on)) < 0:
            raise ValueError("An operator acts on an invalid set of sites.")

        for i, aon in enumerate(acting_on):
            if len(aon) != len(set(aon)):
                raise ValueError(
                    f"The operator at index {i} acts on duplicated sites {aon}"
                )

    acting_on = [tuple(aon) for aon in acting_on]
    # operators = [np.asarray(operator) for operator in operators]
    operators = [_standardize_matrix_input_type(op) for op in operators]

    dtype = canonicalize_dtypes(float, *operators, constant, dtype=dtype)

    canonicalized_operators = []
    canonicalized_acting_on = []
    for operator, acting_on in zip(operators, acting_on):
        check_valid_opmatrix(hilbert, operator, acting_on)

        if operator.dtype is not dtype:
            operator = cast_operator_matrix_dtype(operator, dtype=dtype)

        # re-sort the operator
        operator, acting_on = _reorder_kronecker_product(hilbert, operator, acting_on)
        canonicalized_operators.append(operator)
        canonicalized_acting_on.append(acting_on)

    return canonicalized_operators, canonicalized_acting_on, dtype


def check_valid_opmatrix(hi, mat, acting_on):
    """ """
    expected_size = np.prod([hi.shape[aon] for aon in acting_on])

    if mat.shape != (expected_size, expected_size):
        raise ValueError(
            f"The matrix of the sub-operator acting on sites {acting_on} "
            f"must have shape {expected_size, expected_size}, "
            f"but it has shape {mat.shape}."
        )


# TODO: support sparse arrays without returning dense arrays
def _reorder_kronecker_product(hi, mat, acting_on) -> tuple[Array, tuple]:
    """
    Reorders the matrix resulting from a kronecker product of several
    operators in such a way to sort acting_on.

    A conceptual example is the following:
    if `mat = Â ⊗ B̂ ⊗ Ĉ` and `acting_on = [[2],[1],[3]`
    you will get `result = B̂ ⊗ Â ⊗ Ĉ, [[1], [2], [3]].

    However, essentially, A,B,C represent some operators acting on
    thei sub-space acting_on[1], [2] and [3] of the hilbert space.

    This function also handles any possible set of values in acting_on.

    The inner logic uses the Fock.all_states(), number_to_state and
    state_to_number to perform the re-ordering.
    """
    acting_on_sorted = np.sort(acting_on)
    if np.array_equal(acting_on_sorted, acting_on):
        return mat, acting_on

    # could write custom binary <-> int logic instead of using Fock...
    # Since i need to work with bit-strings (where instead of bits i
    # have integers, in order to support arbitrary size spaces) this
    # is exactly what hilbert.to_number() and vice versa do.

    # target ordering binary representation
    hi_subspace = Fock(hi.shape[acting_on_sorted[0]] - 1)
    for site in acting_on_sorted[1:]:
        hi_subspace = hi_subspace * Fock(hi.shape[site] - 1)

    hi_unsorted_subspace = Fock(hi.shape[acting_on[0]] - 1)
    for site in acting_on[1:]:
        hi_unsorted_subspace = hi_unsorted_subspace * Fock(hi.shape[site] - 1)

    # find how to map target ordering back to unordered
    acting_on_unsorted_ids = np.zeros(len(acting_on), dtype=np.intp)
    for i, site in enumerate(acting_on):
        acting_on_unsorted_ids[i] = np.argmax(site == acting_on_sorted)

    # now it is valid that
    # acting_on_sorted == acting_on[acting_on_unsorted_ids]

    # generate n-bit strings in the target ordering
    v = hi_subspace.all_states()

    # convert them to origin (unordered) ordering
    v_unsorted = v[:, acting_on_unsorted_ids]
    # convert the unordered bit-strings to numbers in the target space.
    n_unsorted = hi_unsorted_subspace.states_to_numbers(v_unsorted)

    # reorder the matrix
    if sparse.issparse(mat):
        # covert the indices from going from old -> new to new->old
        idxs = np.argsort(n_unsorted)

        # sparse matrices don't support indexing, and can be sorted more
        # quickly
        mat_sorted = mat.copy()
        mat_sorted.row[:] = idxs[mat.row]
        mat_sorted.col[:] = idxs[mat.col]
    else:
        mat_sorted = mat[n_unsorted, :][:, n_unsorted]

    return mat_sorted, tuple(acting_on_sorted)


# TODO: support sparse arrays without returning dense arrays
def _multiply_operators(
    hilbert, support_A: tuple, A: Array, support_B: tuple, B: Array, *, dtype
) -> tuple[tuple, Array]:
    """
    Returns the `Tuple[acting_on, Matrix]` representing the operator obtained by
    multiplying the two input operators A and B.
    """
    support_A = np.asarray(support_A)
    support_B = np.asarray(support_B)

    inters = np.intersect1d(support_A, support_B, return_indices=False)

    if support_A.size == support_B.size and np.array_equal(support_A, support_B):
        # TODO: Note: COO @ COO -> CSR, this may insert CSR matrices in our
        # format. Should we enforce COO?
        # If yes, the logic in the compilation can be simplified to always assume
        # COO.
        return tuple(support_A), A @ B
    elif inters.size == 0:
        # disjoint supports
        support = tuple(np.concatenate([support_A, support_B]))
        operator = _kron(A, B)
        operator, support = _reorder_kronecker_product(hilbert, operator, support)
        return tuple(support), operator
    else:
        _support_A = list(support_A)
        _support_B = list(support_B)
        _A = A.copy()
        _B = B.copy()

        # expand _act to match _act_i
        supp_B_min = min(support_B)
        for site in support_A:
            if site not in support_B:
                I = _eye_like(hilbert.shape[site], dtype=dtype, like=B)
                if site < supp_B_min:
                    _support_B = [site, *_support_B]
                    _B = _kron(I, _B)
                else:  # site > actmax
                    _support_B = [*_support_B, site]
                    _B = _kron(_B, I)

        supp_A_min = min(support_A)
        for site in support_B:
            if site not in support_A:
                I = _eye_like(hilbert.shape[site], dtype=dtype, like=A)
                if site < supp_A_min:
                    _support_A = [site, *_support_A]
                    _A = _kron(I, _A)
                else:  # site > actmax
                    _support_A = [*_support_A, site]
                    _A = _kron(_A, I)

        # reorder
        _A, _support_A = _reorder_kronecker_product(hilbert, _A, _support_A)
        _B, _support_B = _reorder_kronecker_product(hilbert, _B, _support_B)

        if len(_support_A) == len(_support_B) and np.array_equal(
            _support_A, _support_B
        ):
            # back to the case of non-intersecting with same support
            # TODO: Note: COO @ COO -> CSR
            return tuple(_support_A), _A @ _B
        else:
            raise ValueError("Something failed")
