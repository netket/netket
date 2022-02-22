import abc
from typing import Tuple
import numpy as np
import jax.numpy as jnp

from numba import jit
from scipy.sparse import csr_matrix as _csr_matrix

from netket.hilbert import DiscreteHilbert
from netket.operator import AbstractOperator


class DiscreteOperator(AbstractOperator):
    r"""This class is the base class for operators defined on a
    discrete Hilbert space. Users interested in implementing new
    quantum Operators for discrete Hilbert spaces should derive
    their own class from this class
    """

    def __init__(self, hilbert: DiscreteHilbert):
        if not isinstance(hilbert, DiscreteHilbert):
            raise ValueError(
                "A Discrete Operator can only act upon a discrete Hilbert space."
            )
        super().__init__(hilbert)

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        raise NotImplementedError

    def get_conn_padded(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator.

        Starting from a batch of quantum numbers :math:`x={x_1, ... x_n}` of
        size :math:`B \times M` where :math:`B` size of the batch and :math:`M`
        size of the hilbert space, finds all states :math:`y_i^1, ..., y_i^K`
        connected to every :math:`x_i`.

        Returns a matrix of size :math:`B \times K_{max} \times M` where
        :math:`K_{max}` is the maximum number of connections for every
        :math:`y_i`.

        Args:
            x : A N-tensor of shape :math:`(...,hilbert.size)` containing
                the batch/batches of quantum numbers :math:`x`.
        Returns:
            **(x_primes, mels)**: The connected states x', in a N+1-tensor and an
            N-tensor containing the matrix elements :math:`O(x,x')`
            associated to each x' for every batch.
        """
        n_visible = x.shape[-1]
        n_samples = x.size // n_visible

        sections = np.empty(n_samples, dtype=np.int32)
        x_primes, mels = self.get_conn_flattened(
            x.reshape(-1, x.shape[-1]), sections, pad=True
        )

        n_primes = sections[0]

        x_primes_r = x_primes.reshape(*x.shape[:-1], n_primes, n_visible)
        mels_r = mels.reshape(*x.shape[:-1], n_primes)

        return x_primes_r, mels_r

    @abc.abstractmethod
    def get_conn_flattened(
        self, x: np.ndarray, sections: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator.

        Starting from a given quantum number :math:`x`, it finds all
        other quantum numbers  :math:`x'` such that the matrix element
        :math:`O(x,x')` is different from zero. In general there will be
        several different connected states :math:`x'` satisfying this
        condition, and they are denoted here :math:`x'(k)`, for
        :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape
        :math:`(batch_size,hilbert.size)`.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size)
                containing the batch of quantum numbers x.
            sections (array): An array of sections for the flattened x'.
                See numpy.split for the meaning of sections.

        Returns:
            (matrix, array): The connected states x', flattened together in
                a single matrix.
                An array containing the matrix elements :math:`O(x,x')`
                associated to each x'.

        """

    def get_conn(self, x: np.ndarray):
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
        Raise:
            ValueError: If the given quantum number is not compatible with the hilbert space.
        """
        if x.ndim != 1:
            raise ValueError(
                "get_conn does not support batches. Please use get_conn_flattened instead."
            )
        if x.shape[0] != self.hilbert.size:
            raise ValueError(
                "The given quantum numbers do not match the hilbert space."
            )

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

    def to_qobj(self) -> "qutip.Qobj":  # noqa: F821
        r"""Convert the operator to a qutip's Qobj.

        Returns:
            A :ref:`qutip.Qobj` object.
        """
        from qutip import Qobj

        return Qobj(
            self.to_sparse(), dims=[list(self.hilbert.shape), list(self.hilbert.shape)]
        )

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.apply(v)

    def apply(self, v: np.ndarray) -> np.ndarray:
        op = self.to_linear_operator()
        return op.dot(v)

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
