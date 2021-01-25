import abc
import numpy as _np
from scipy.sparse import csr_matrix as _csr_matrix
from numba import jit


class AbstractOperator(abc.ABC):
    """Abstract class for quantum Operators. This class prototypes the methods
    needed by a class satisfying the Operator concept. Users interested in
    implementing new quantum Operators should derive they own class from this
    class
    """

    @property
    @abc.abstractmethod
    def size(self):
        r"""int: The total number number of local degrees of freedom."""
        raise NotImplementedError()

    def get_conn_padded(self, x):
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
        sections = _np.empty(x.shape[0], dtype=_np.int32)
        x_primes, mels = self.get_conn_flattened(x, sections, pad=True)

        n_primes = sections[0]
        n_visible = x.shape[1]

        x_primes_r = x_primes.reshape(-1, n_primes, n_visible)
        mels_r = mels.reshape(-1, n_primes)

        return x_primes_r, mels_r

    @abc.abstractmethod
    def get_conn_flattened(self, x, sections):
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

    def n_conn(self, x, out=None):
        r"""Return the number of states connected to x.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            out (array): If None an output array is allocated.

        Returns:
            array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.intc)
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

    @property
    @abc.abstractmethod
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        raise NotImplementedError()

    @property
    def is_hermitian(self):
        return False

    def to_sparse(self):
        r"""Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            scipy.sparse.csr_matrix: The sparse matrix representation of the operator.
        """
        hilb = self.hilbert

        x = hilb.all_states()

        sections = _np.empty(x.shape[0], dtype=_np.int32)
        x_prime, mels = self.get_conn_flattened(x, sections)

        numbers = hilb.states_to_numbers(x_prime)

        sections1 = _np.empty(sections.size + 1, dtype=_np.int32)
        sections1[1:] = sections
        sections1[0] = 0

        return _csr_matrix((mels, numbers, sections1))

    def to_dense(self):
        r"""Returns the dense matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            numpy.ndarray: The dense matrix representation of the operator as a Numpy array.
        """
        return self.to_sparse().todense().A

    def apply(self, v):
        return self.to_linear_operator().dot(v)

    def __call__(self, v):
        return self.apply(v)

    def to_linear_operator(self):
        return self.to_sparse()

    def __repr__(self):
        return f"{type(self).__name__}(hilbert={self.hilbert})"
