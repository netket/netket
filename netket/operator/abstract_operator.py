import abc


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
        return NotImplementedError

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
        return NotImplementedError

    @abc.abstractmethod
    def n_conn(self, x, out=None):
        r"""Return the number of states connected to x.

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                out (array): If None an output array is allocated.            

            Returns:
                array: The number of connected states x' for each x[i].

        """
        return NotImplementedError

    @property
    @abc.abstractmethod
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return NotImplementedError
