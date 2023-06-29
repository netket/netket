import jax
import jax.numpy as jnp
import numpy as np
import itertools
from scipy.sparse import issparse

import netket as nk

# create some globals
I = jnp.eye(2)
X = jnp.array([[0, 1], [1, 0]])
Y = jnp.array([[0, -1j], [1j, 0]])
Z = jnp.array([[1, 0], [0, -1]])
pauli_basis = jnp.stack([I, X, Y, Z], axis=0)
pauli_basis_str = list("IXYZ")


@jax.jit
def _tensor_product(mats1, mats2):
    """Get all combinations between two sets via tensorproduct"""
    if mats1.ndim == 2 and mats2.ndim == 2:
        return jnp.kron(mats1, mats2)
    elif mats1.ndim == 3:
        return jax.vmap(_tensor_product, in_axes=(0, None))(mats1, mats2)
    elif mats2.ndim == 3:
        return jax.vmap(_tensor_product, in_axes=(None, 0))(mats1, mats2)


def _get_basis_till_n(n=1):
    """Get the basis set till size 2^n x 2^n size

    Returns a dictionary:
    every value is a basis set (# basis elements) x 2^n x 2^n
    the keys is 2**n
    """
    if n == 1:
        return {2: pauli_basis}
    else:
        lower_bases = _get_basis_till_n(n=n - 1)
        tp = _tensor_product(pauli_basis, lower_bases[2 ** (n - 1)])
        tp = tp.reshape(-1, *tp.shape[-2:])
        return {**lower_bases, (2**n): tp}


@jax.jit
def _hilbert_schmidt(A, basis_element):
    """Hilbert schmidt product where the basis elements are hermitian"""
    return jnp.trace(basis_element @ A)


def _make_basis_strings(n_acting):
    """Make the strings corresponding to the tensorproduct of Pauli operators"""
    bs = list(itertools.product(pauli_basis_str, repeat=n_acting))
    bs = list(map(lambda b: "".join(b), bs))
    return np.array(bs)


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


def local_operators_to_pauli_strings(local_op):
    """Convert a LocalOperator into PauliStrings

    Args:
        local_op: LocalOperator

    Returns:
        PauliStrings
    """
    mats = local_op.operators
    acting_on = local_op.acting_on
    constant = local_op.constant
    dtype = local_op.dtype
    hi = local_op.hilbert

    operators = []
    weights = []

    if len(mats) > 0:

        def _convert_to_dense(m):
            return m.todense() if issparse(m) else m

        mats = list(map(_convert_to_dense, mats))
        mats_sizes = list(map(len, mats))
        max_size = max(mats_sizes)
        n_max = int(np.log2(max_size))
        unique_mat_sizes = np.unique(mats_sizes)

        # make the basis and the string representation
        bases = _get_basis_till_n(n=n_max)
        bases_strs = {
            size: _make_basis_strings(int(np.log2(size))) for size in unique_mat_sizes
        }

        # loop trough operators and convert to Pauli strings by projecting
        for op, act, size in zip(mats, acting_on, mats_sizes):
            op_list, w_list = _local_operator_to_pauli_strings(
                op, act, bases[size], bases_strs[size], hi.size
            )
            operators += op_list
            weights += w_list

        # add the constant
        operators.append("I" * hi.size)
        weights.append(constant)

    return nk.operator.PauliStrings(
        hi, operators=operators, weights=weights, dtype=dtype
    )
