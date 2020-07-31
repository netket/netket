from ._abstract_operator import AbstractOperator
from ..hilbert import Boson

import math as _m
import numpy as _np
from numba import jit


class BoseHubbard(AbstractOperator):
    r"""
    An extended Bose Hubbard model Hamiltonian operator, containing both
    on-site interactions and nearest-neighboring density-density interactions.
    """

    def __init__(self, hilbert, U, V=0, J=1, mu=0):
        r"""
        Constructs a new ``BoseHubbard`` given a hilbert space and a Hubbard
        interaction strength. The chemical potential and the density-density interaction strenght
        can be specified as well.

        Args:
           hilbert (netket.hilbert.Boson): Hilbert space the operator acts on.
           U (float): The Hubbard interaction term.
           V (float): The strenght of density-density interaction term.
           J (float): The hopping amplitude.
           mu (float): The chemical potential.

        Examples:
           Constructs a ``BoseHubbard`` operator for a 2D system.

           >>> import netket as nk
           >>> g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
           >>> hi = nk.hilbert.Boson(n_max=3, n_bosons=6, graph=g)
           >>> op = nk.operator.BoseHubbard(U=4.0, hilbert=hi)
           >>> print(op.hilbert.size)
           9
        """
        self._U = U
        self._V = V
        self._J = J
        self._mu = mu
        self._hilbert = hilbert
        assert isinstance(hilbert, Boson)

        self._n_max = hilbert.n_max
        self._n_sites = hilbert.size
        self._edges = _np.asarray(hilbert.graph.edges())
        self._max_conn = 1 + self._edges.shape[0] * 2
        self._max_mels = _np.empty(self._max_conn, dtype=_np.complex128)
        self._max_xprime = _np.empty((self._max_conn, self._n_sites))

        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self):
        return self._hilbert.size

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
        mels = self._max_mels
        x_prime = self._max_xprime

        mels[0] = 0.0
        x_prime[0] = _np.copy(x)

        J = self._J
        V = self._V
        sqrt = _m.sqrt
        n_max = self._n_max

        c = 1
        for e in self._edges:
            i, j = e
            n_i = x[i]
            n_j = x[j]
            mels[0] += V * n_i * n_j

            # destroy on i create on j
            if n_i > 0 and n_j < n_max:
                mels[c] = -J * sqrt(n_i) * sqrt(n_j + 1)
                x_prime[c] = _np.copy(x)
                x_prime[c, i] -= 1.0
                x_prime[c, j] += 1.0
                c += 1

            # destroy on j create on i
            if n_j > 0 and n_i < n_max:
                mels[c] = -J * sqrt(n_j) * sqrt(n_i + 1)
                x_prime[c] = _np.copy(x)
                x_prime[c, j] -= 1.0
                x_prime[c, i] += 1.0
                c += 1

        mu = self._mu
        Uh = 0.5 * self._U
        for i in range(self._n_sites):
            # chemical potential
            mels[0] -= mu * x[i]
            # on-site interaction
            mels[0] += Uh * x[i] * (x[i] - 1.0)

        return _np.copy(x_prime[:c]), _np.copy(mels[:c])

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(
        x, sections, edges, mels, x_prime, U, V, J, mu, n_max, max_conn
    ):

        batch_size = x.shape[0]
        n_sites = x.shape[1]

        if mels.size < batch_size * max_conn:
            mels = _np.empty(batch_size * max_conn, dtype=_np.complex128)
            x_prime = _np.empty((batch_size * max_conn, n_sites))

        sqrt = _m.sqrt
        Uh = 0.5 * U

        diag_ind = 0
        for b in range(batch_size):
            mels[diag_ind] = 0.0
            x_prime[diag_ind] = _np.copy(x[b])

            for i in range(n_sites):
                # chemical potential
                mels[diag_ind] -= mu * x[b, i]
                # on-site interaction
                mels[diag_ind] += Uh * x[b, i] * (x[b, i] - 1.0)

            odiag_ind = 1 + diag_ind
            for e in range(edges.shape[0]):
                i, j = edges[e][0], edges[e][1]
                n_i = x[b, i]
                n_j = x[b, j]
                mels[diag_ind] += V * n_i * n_j

                # destroy on i create on j
                if n_i > 0 and n_j < n_max:
                    mels[odiag_ind] = -J * sqrt(n_i) * sqrt(n_j + 1)
                    x_prime[odiag_ind] = _np.copy(x[b])
                    x_prime[odiag_ind, i] -= 1.0
                    x_prime[odiag_ind, j] += 1.0
                    odiag_ind += 1

                # destroy on j create on i
                if n_j > 0 and n_i < n_max:
                    mels[odiag_ind] = -J * sqrt(n_j) * sqrt(n_i + 1)
                    x_prime[odiag_ind] = _np.copy(x[b])
                    x_prime[odiag_ind, j] -= 1.0
                    x_prime[odiag_ind, i] += 1.0
                    odiag_ind += 1

            diag_ind = odiag_ind

            sections[b] = odiag_ind

        return _np.copy(x_prime[:odiag_ind]), _np.copy(mels[:odiag_ind])

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
        return self._flattened_kernel(
            x,
            sections,
            self._edges,
            self._max_mels,
            self._max_xprime,
            self._U,
            self._V,
            self._J,
            self._mu,
            self._n_max,
            self._max_conn,
        )
