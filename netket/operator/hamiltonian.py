from .._C_netket.operator import GraphOperator as _GraphOperator
from .abstract_operator import AbstractOperator
from .graph_operator import PyGraphOperator

import numpy as _np
from numba import jit


def Ising(hilbert, h, J=1.0):
    """
    Constructs a new ``Ising`` given a hilbert space, a transverse field,
    and (if specified) a coupling constant.

    Args:
        hilbert: Hilbert space the operator acts on.
        h: The strength of the transverse field.
        J: The strength of the coupling. Default is 1.0.

    Examples:
        Constructs an ``Ising`` operator for a 1D system.

        >>> import netket as nk
        >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
        >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
        >>> op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5)
        >>> print(op.hilbert.size)
        20
    """
    sigma_x = _np.array([[0, 1], [1, 0]])
    sz_sz = _np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                       [0, 0, -1, 0], [0, 0, 0, 1]])
    return _GraphOperator(hilbert, siteops=[-h * sigma_x], bondops=[J * sz_sz])


def Heisenberg(hilbert, J=1, sign_rule=None):
    """
    Constructs a new ``Heisenberg`` given a hilbert space.

    Args:
        hilbert: Hilbert space the operator acts on.
        J: The strength of the coupling. Default is 1.
        sign_rule: If enabled, Marshal's sign rule will be used. On a bipartite
                   lattice, this corresponds to a basis change flipping the Sz direction
                   at every odd site of the lattice. For non-bipartite lattices, the
                   sign rule cannot be applied. Defaults to True if the lattice is
                   bipartite, False otherwise.

    Examples:
     Constructs a ``Heisenberg`` operator for a 1D system.

        >>> import netket as nk
        >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
        >>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
        >>> op = nk.operator.Heisenberg(hilbert=hi)
        >>> print(op.hilbert.size)
        20
    """
    if sign_rule is None:
        sign_rule = hilbert.graph.is_bipartite

    sz_sz = _np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                       [0, 0, -1, 0], [0, 0, 0, 1]])
    exchange = _np.array(
        [[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    if sign_rule:
        if not hilbert.graph.is_bipartite:
            raise ValueError(
                "sign_rule=True specified for a non-bipartite lattice")
        heis_term = sz_sz - exchange
    else:
        heis_term = sz_sz + exchange
    return _GraphOperator(hilbert, bondops=[J * heis_term])


class PyIsing(AbstractOperator):

    def __init__(self, hilbert, h, J=1.0):
        """
        Constructs a new ``Ising`` given a hilbert space, a transverse field,
        and (if specified) a coupling constant.

        Args:
            hilbert: Hilbert space the operator acts on.
            h: The strength of the transverse field.
            J: The strength of the coupling. Default is 1.0.

        Examples:
            Constructs an ``Ising`` operator for a 1D system.

            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5)
            >>> print(op.hilbert.size)
            20
        """
        self._h = h
        self._J = J
        self._hilbert = hilbert
        self._n_sites = hilbert.size
        self._section = hilbert.size + 1
        self._edges = _np.asarray(hilbert.graph.edges)
        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self):
        return self._n_sites

    def n_conn(self, x, out):
        r"""Return the number of states connected to x.

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                out (array): If None an output array is allocated.

            Returns:
                array: The number of connected states x' for each x[i].

        """
        if(out is None):
            out = _np.empty(x.shape[0], dtype=_np.int32)

        out.fill(hilbert.size + 1)

        return out

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
            from a given quantum number x, it finds all other quantum numbers x' such
            that the matrix element :math:`O(x,x')` is different from zero. In general there
            will be several different connected states x' satisfying this
            condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

            This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

            Args:
                x (array): An array of shape (hilbert.size) containing the quantum numbers x.

            Returns:
                matrix: The connected states x' of shape (N_connected,hilbert.size)
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        mels = _np.empty(x.shape[0] + 1, dtype=_np.complex128)
        x_prime = _np.tile(x, (x.shape[0] + 1, 1))

        mels[0] = 0.
        for e in self._edges:
            mels[0] += self._J * (x[e[0]] * x[e[1]])

        mels[1:].fill(-self._h)

        for i in range(x.shape[0]):
            x_prime[i + 1][i] *= -1.

        return x_prime, mels

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(x, sections, edges, h, J):
        n_sites = x.shape[1]
        n_conn = n_sites + 1

        x_prime = _np.empty((x.shape[0] * n_conn, n_sites))
        mels = _np.empty(x.shape[0] * n_conn)

        diag_ind = 0

        for i in range(x.shape[0]):

            mels[diag_ind] = 0.
            for k in range(edges.shape[0]):
                mels[diag_ind] += J * x[i, edges[k, 0]] * x[i, edges[k, 1]]

            odiag_ind = 1 + diag_ind

            mels[odiag_ind:(odiag_ind + n_sites)].fill(-h)

            x_prime[diag_ind:(diag_ind + n_conn)] = x[i]

            for j in range(n_sites):
                x_prime[j + odiag_ind][j] *= -1.

            diag_ind += n_conn

            sections[i] = diag_ind

        return x_prime, mels

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
                sections (array): An array of size (batch_size) useful to unflatten
                            the output of this function.
                            See numpy.split for the meaning of sections.

            Returns:
                matrix: The connected states x', flattened together in a single matrix.
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._flattened_kernel(x, sections, self._edges, self._h, self._J)


def PyHeisenberg(hilbert, J=1, sign_rule=None):
    """
    Constructs a new ``Heisenberg`` given a hilbert space.

    Args:
        hilbert: Hilbert space the operator acts on.
        J: The strength of the coupling. Default is 1.
        sign_rule: If enabled, Marshal's sign rule will be used. On a bipartite
                   lattice, this corresponds to a basis change flipping the Sz direction
                   at every odd site of the lattice. For non-bipartite lattices, the
                   sign rule cannot be applied. Defaults to True if the lattice is
                   bipartite, False otherwise.

    Examples:
     Constructs a ``Heisenberg`` operator for a 1D system.

        >>> import netket as nk
        >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
        >>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
        >>> op = nk.operator.Heisenberg(hilbert=hi)
        >>> print(op.hilbert.size)
        20
    """
    if sign_rule is None:
        sign_rule = hilbert.graph.is_bipartite

    sz_sz = _np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                       [0, 0, -1, 0], [0, 0, 0, 1]])
    exchange = _np.array(
        [[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
    if sign_rule:
        if not hilbert.graph.is_bipartite:
            raise ValueError(
                "sign_rule=True specified for a non-bipartite lattice")
        heis_term = sz_sz - exchange
    else:
        heis_term = sz_sz + exchange
    return PyGraphOperator(hilbert, bondops=[J * heis_term])
