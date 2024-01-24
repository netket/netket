# Copyright 2023 The NetKet Authors - All rights reserved.
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


import jax
import jax.numpy as jnp
import numpy as np
import itertools
from scipy.sparse import issparse

from netket.operator import PauliStrings

# Pauli Matrices: shape (2, 2)
I = jnp.eye(2)
X = jnp.array([[0, 1], [1, 0]])
Y = jnp.array([[0, -1j], [1j, 0]])
Z = jnp.array([[1, 0], [0, -1]])

# stacked pauli matrices: shape (4, 2, 2)
pauli_basis = jnp.stack([I, X, Y, Z], axis=0)
pauli_basis_str = list("IXYZ")


@jax.jit
def _tensor_product(mats1, mats2):
    """Get all combinations of Pauli operators between two sets
    via tensorproduct.

    Args:
        mats1: either a (2ⁿ,2ⁿ) representing a single operator or a (N,2ⁿ,2ⁿ)
            tensor encoding N different pauli operators
        mats2: either a (2ᵐ,2ᵐ) representing a single operator or a (M,2ᵐ,2ᵐ)
            tensor encoding M different pauli operators
    Returns
        A (2ⁿ⁺ᵐ, 2ⁿ⁺ᵐ) tensor if both mats1 and mats2 are single paulis. Otherwise
        an (N,M, 2ⁿ⁺ᵐ, 2ⁿ⁺ᵐ) tensor (where N or M dimension might be dropped if
        not there in the input).
    """
    if mats1.ndim == 2 and mats2.ndim == 2:
        return jnp.kron(mats1, mats2)
    elif mats1.ndim == 3:
        return jax.vmap(_tensor_product, in_axes=(0, None))(mats1, mats2)
    elif mats2.ndim == 3:
        return jax.vmap(_tensor_product, in_axes=(None, 0))(mats1, mats2)
    else:
        raise ValueError("Not supposed to get more than 3 dimensions")


def _get_basis_till_n(n=1):
    """Get the basis set till the basis set with at most n non
    identities in the Pauli Strings, which will contain
    4ⁿ elements of size (2ⁿ, 2ⁿ).

    Returns:
        A dictionary with keys [1, 2, ..., n] where every value is
        a basis set encoded into a tensor of shape (4ⁿ, 2ⁿ, 2ⁿ)
        where the first dimension (4ⁿ) is the number of basis elements
        for that number of non-identity paulis, and 2ⁿ is the size
        of those matrices.

        The key represents the size of those matrices.
    """
    if n == 1:
        return {n: pauli_basis}
    else:
        lower_bases = _get_basis_till_n(n=n - 1)
        # extract the basis for n-1, which is (4**(n-1), 2ⁿ, 2ⁿ) tensor
        nm1_bases = lower_bases[n - 1]

        # compute the (4,4ⁿ⁻¹, 4, 4)
        tp = _tensor_product(pauli_basis, nm1_bases)

        # reshape everything to a single set of bases (4ⁿ, ...)
        tp = tp.reshape(-1, *tp.shape[-2:])
        return {**lower_bases, n: tp}


def _make_basis_strings(n_acting):
    """Make the strings corresponding to the tensorproduct of Pauli operators"""
    bs = list(itertools.product(pauli_basis_str, repeat=n_acting))
    bs = list(map(lambda b: "".join(b), bs))
    return np.array(bs)


def _get_paulistrings_till_n(n=1):
    """
    Get the
    """
    return {n_paulis: _make_basis_strings(n_paulis) for n_paulis in range(1, n + 1)}


@jax.jit
def _hilbert_schmidt(A, basis_element):
    """Hilbert schmidt product where the basis elements are hermitian"""
    return jnp.trace(basis_element @ A)


@jax.jit
def _project_matrix(A, basis_set):
    """
    Project matrix onto basis of tensorproduct of Paulis
    according to hilbert-schmidt product
    """
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    size = A.shape[0]
    return jax.vmap(_hilbert_schmidt, in_axes=(None, 0))(A, basis_set) / size


def _create_pauli_string(substr, acting_on, hilbert_size):
    """Create a single PauliString from the operators and indices"""
    s = list("I" * hilbert_size)
    for i, si in zip(acting_on, substr):
        s[i] = si
    s = "".join(list(s))
    return s


def _local_operator_to_pauli_strings(
    op, acting_on, basis_set, basis_set_str, hilbert_size
):
    """Convert a matrix into a sum of Pauli strings"""
    acting_on = np.array(acting_on)
    all_weights = _project_matrix(op, basis_set)
    nz_idxs = np.nonzero(all_weights)[0]
    weights = all_weights[nz_idxs]

    # the next is not efficient
    operators = [basis_set_str[idx] for idx in nz_idxs]
    _str_maker = lambda s: _create_pauli_string(s, acting_on, hilbert_size)
    operators = list(map(_str_maker, operators))
    return list(operators), list(weights)


def _convert_to_dense(m):
    return m.todense() if issparse(m) else m


def local_operators_to_pauli_strings(hilbert, operators, acting_on, constant, dtype):
    """Convert a LocalOperator into PauliStrings

    Args:
        local_op: LocalOperator

    Returns:
        PauliStrings
    """

    if any(d != 2 for d in hilbert.shape):
        raise TypeError(
            "Cannot convert to Pauli strings operators defined on hilbert spaces "
            "with local dimension != 2"
        )

    pauli_strings = []
    weights = []

    if len(operators) > 0:
        # maximum number of non-identity operators
        n_max = max(list(map(len, acting_on)))

        # make the basis and the string representation
        bases = _get_basis_till_n(n=n_max)
        bases_strs = _get_paulistrings_till_n(n=n_max)

        # loop trough operators and convert to Pauli strings by projecting
        for op, act in zip(operators, acting_on):
            # convert to dense-numpy as we later use jax which does not support sparse.
            op = _convert_to_dense(op)

            n_paulis = len(act)

            op_list, w_list = _local_operator_to_pauli_strings(
                op, act, bases[n_paulis], bases_strs[n_paulis], hilbert.size
            )
            pauli_strings += op_list
            weights += w_list

    # add the constant if it's not zero
    if np.abs(constant) > 1e-13:
        pauli_strings.append("I" * hilbert.size)
        weights.append(constant)

    # the calculation above returns complex weights even for operators that are
    # purely real, so we discard their imaginary parts
    weights = [x.real if np.isreal(x) else x for x in weights]

    res = PauliStrings(hilbert, operators=pauli_strings, weights=weights, dtype=dtype)
    return res
