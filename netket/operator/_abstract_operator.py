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

import abc
from typing import Optional, Tuple, List

import numpy as np
import jax.numpy as jnp
from netket.utils.types import DType

from scipy.sparse import csr_matrix as _csr_matrix
from numba import jit

from netket.hilbert import AbstractHilbert


@jit(nopython=True)
def compute_row_indices(rows, sections):
    ntot = sections[-1]
    res = np.empty(ntot, dtype=np.intp)

    for i in range(1, sections.size):
        res[sections[i - 1] : sections[i]] = rows[i - 1]

    return res


class AbstractOperator(abc.ABC):
    """Abstract class for quantum Operators. This class prototypes the methods
    needed by a class satisfying the Operator concept. Users interested in
    implementing new quantum Operators should derive they own class from this
    class
    """

    _hilbert: AbstractHilbert
    r"""The hilbert space associated to this operator."""

    def __init__(self, hilbert: AbstractHilbert):
        self._hilbert = hilbert

    @property
    def hilbert(self) -> AbstractHilbert:
        r"""The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self) -> int:
        r"""The total number number of local degrees of freedom."""
        return self._hilbert.size

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        return False

    @property
    def H(self) -> "AbstractOperator":
        """Returns the Conjugate-Transposed operator"""
        if self.is_hermitian:
            return self

        from ._lazy import Adjoint

        return Adjoint(self)

    @property
    def T(self) -> "AbstractOperator":
        """Returns the transposed operator"""
        return self.transpose()

    @property
    @abc.abstractmethod
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        raise NotImplementedError

    def collect(self) -> "AbstractOperator":
        """
        Returns a guranteed concrete instancce of an operator.

        As some operations on operators return lazy wrapperes (such as transpose,
        hermitian conjugate...), this is used to obtain a guaranteed non-lazy
        operator.
        """
        return self

    def transpose(self, *, concrete=False) -> "AbstractOperator":
        """Returns the transpose of this operator.

        Args:
            concrete: if True returns a concrete operator and not a lazy wrapper

        Returns:
            if concrete is not True, self or a lazy wrapper; the
            transposed operator otherwise
        """
        if not concrete:
            from ._lazy import Transpose

            return Transpose(self)
        else:
            raise NotImplementedError

    def conjugate(self, *, concrete=False) -> "AbstractOperator":
        """Returns the complex-conjugate of this operator.

        Args:
            concrete: if True returns a concrete operator and not a lazy wrapper

        Returns:
            if concrete is not True, self or a lazy wrapper; the
            complex-conjugated operator otherwise
        """
        raise NotImplementedError

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        raise NotImplementedError

    def get_conn_padded(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator.
        Starting from a batch of quantum numbers x={x_1, ... x_n} of size B x M
        where B size of the batch and M size of the hilbert space, finds all states
        y_i^1, ..., y_i^K connected to every x_i.
        Returns a matrix of size B x Kmax x M where Kmax is the maximum number of
        connections for every y_i.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.

        Returns:
            matrix: The connected states x', in a 3D tensor.
            array: A matrix containing the matrix elements :math:`O(x,x')` associated to each x' for every batch.
        """
        sections = np.empty(x.shape[0], dtype=np.int32)
        x_primes, mels = self.get_conn_flattened(x, sections, pad=True)

        n_primes = sections[0]
        n_visible = x.shape[1]

        x_primes_r = x_primes.reshape(-1, n_primes, n_visible)
        mels_r = mels.reshape(-1, n_primes)

        return x_primes_r, mels_r

    @abc.abstractmethod
    def get_conn_flattened(
        self, x: np.ndarray, sections: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of sections for the flattened x'.
                        See numpy.split for the meaning of sections.

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        raise NotImplementedError()

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        Args:
            x (array): An array of shape (hilbert.size) containing the quantum numbers x.

        Returns:
            matrix: The connected states x' of shape (N_connected,hilbert.size)
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self.get_conn_flattened(
            x.reshape((1, -1)),
            np.ones(1),
        )

    def n_conn(self, x, out=None) -> np.ndarray:
        r"""Return the number of states connected to x.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            out (array): If None an output array is allocated.

        Returns:
            array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = np.empty(x.shape[0], dtype=np.intc)
        self.get_conn_flattened(x, out)
        out = self._n_conn_from_sections(out)

        return out

    @staticmethod
    @jit(nopython=True)
    def _n_conn_from_sections(out):
        low = 0
        for i in range(out.shape[0]):
            old_out = out[i]
            out[i] = out[i] - low
            low = old_out

        return out

    def to_sparse(self) -> _csr_matrix:
        r"""Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The sparse matrix representation of the operator.
        """
        concrete_op = self.collect()
        hilb = self.hilbert

        x = hilb.all_states()

        sections = np.empty(x.shape[0], dtype=np.int32)
        x_prime, mels = concrete_op.get_conn_flattened(x, sections)

        numbers = hilb.states_to_numbers(x_prime)

        sections1 = np.empty(sections.size + 1, dtype=np.int32)
        sections1[1:] = sections
        sections1[0] = 0

        ## eliminate duplicates from numbers
        # rows_indices = compute_row_indices(hilb.states_to_numbers(x), sections1)

        return _csr_matrix(
            (mels, numbers, sections1),
            shape=(self.hilbert.n_states, self.hilbert.n_states),
        )

        # return _csr_matrix(
        #    (mels, (rows_indices, numbers)),
        #    shape=(self.hilbert.n_states, self.hilbert.n_states),
        # )

    def to_dense(self) -> np.ndarray:
        r"""Returns the dense matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The dense matrix representation of the operator as a Numpy array.
        """
        return self.to_sparse().todense().A

    def apply(self, v: np.ndarray) -> np.ndarray:
        op = self.to_linear_operator()
        return op.dot(v)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.apply(v)

    def conj(self, *, concrete=False) -> "AbstractOperator":
        return self.conjugate(concrete=False)

    def to_linear_operator(self):
        return self.to_sparse()

    def _get_conn_flattened_closure(self):
        raise NotImplementedError(
            """
            _get_conn_flattened_closure not implemented for this operator type.
            You were probably trying to use an operator with a sampler.
            Please report this bug.
            
            numba4jax won't work.
            """
        )

    def __repr__(self):
        return f"{type(self).__name__}(hilbert={self.hilbert})"

    def __matmul__(self, other):
        if isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray):
            return self.apply(other)
        elif isinstance(other, AbstractOperator):
            if self == other and self.is_hermitian:
                from ._lazy import Squared

                return Squared(self)
            else:
                return self._op__matmul__(other)
        else:
            return NotImplemented

    def _op__matmul__(self, other):
        "Implementation on subclasses of __matmul__"
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray):
            # return self.apply(other)
            return NotImplemented
        elif isinstance(other, AbstractOperator):
            if self == other and self.is_hermitian:
                from ._lazy import Squared

                return Squared(self)
            else:
                return self._op__rmatmul__(other)
        else:
            return NotImplemented

    def _op__rmatmul__(self, other):
        "Implementation on subclasses of __matmul__"
        return NotImplemented
