from .abstract_operator import AbstractOperator
from .local_operator import PyLocalOperator

import numpy as _np
from numba import jit


class PyGraphOperator(AbstractOperator):
    r"""A graph-based quantum operator. In its simplest terms, this is the sum of
        local operators living on the edge of an arbitrary graph."""

    def __init__(self, hilbert, siteops=[], bondops=[], bondops_colors=[], graph=None):
        r"""
        Constructs a new ``GraphOperator`` given a hilbert space and either a
        list of operators acting on sites or a list acting on the bonds.
        Users can specify the color of the bond that an operator acts on, if
        desired. If none are specified, the bond operators act on all edges.

        Args:
         hilbert: Hilbert space the operator acts on.
         graph: The graph whose vertices and edges are considered to construct the
                operator. If None, the graph is deduced from the hilbert object.
         siteops: A list of operators in matrix form that act
                on the nodes of the graph.
                The default is an empty list. Note that if no siteops are
                specified, the user must give a list of bond operators.
         bondops: A list of operators that act on the edges of the graph.
             The default is None. Note that if no bondops are
             specified, the user must give a list of site operators.
         bondops_colors: A list of edge colors, specifying the color each
             bond operator acts on. The defualt is an empty list.

        Examples:
         Constructs a ``BosGraphOperator`` operator for a 2D system.

         >>> import netket as nk
         >>> sigmax = [[0, 1], [1, 0]]
         >>> mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
         >>> edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
         ... [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
         ... [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]
         >>> g = nk.graph.CustomGraph(edges=edges)
         >>> hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], graph=g)
         >>> op = nk.operator.GraphOperator(
         ... hi, siteops=[sigmax], bondops=[mszsz])
         >>> print(op.hilbert.size)
         20
        """
        self._hilbert = hilbert
        if graph is None:
            graph = self._hilbert.graph

        self._graph = graph
        self._size = graph.size

        # Site operators
        operators = []
        acting_on = []

        if(len(siteops) > 0):
            assert(isinstance(siteops, list))

            for i in range(self._size):
                operators.append(
                    _np.asarray(siteops[0], dtype=_np.complex128))
                for j in range(1, len(siteops), 1):
                    operators[i] += siteops[j]
                acting_on.append([i])

        if(len(bondops) > 0):
            if(len(bondops_colors) > 0):
                raise NotImplementedError(
                    "GraphOperator with bond colors are not implemented yet")
            for edge in graph.edges:
                operators.append(bondops[0])
                acting_on.append(edge)

        self._local_operator = PyLocalOperator(hilbert, operators, acting_on)

        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self):
        return self._size

    def graph(self):
        r"""AbstractGraph: The graph associated to this operator."""
        return self._graph

    def n_conn(self, x, out):
        r"""Return the number of states connected to x.

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                out (array): If None an output array is allocated.

            Returns:
                array: The number of connected states x' for each x[i].

        """
        return self._local_operator.n_conn(x, out)

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
        return self._local_operator.get_conn(x)

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

        return self._local_operator.get_conn_flattened(x, sections)
