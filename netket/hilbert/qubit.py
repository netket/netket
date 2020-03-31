from .abstract_hilbert import AbstractHilbert
from .custom_hilbert import PyCustomHilbert

import numpy as _np
from netket import random as _random
from numba import jit
from netket.graph import Edgeless


class Qubit(PyCustomHilbert):
    r"""Hilbert space obtained as tensor product of local qubit states."""

    def __init__(self, graph=None, size=None):
        r"""Initializes a qubit hilbert space.

        Args:
        graph: Graph representation of qubits. If None, size
              is used to fix the total number of qubits.
        size: Number of qubits. If None, a graph must be speficied.


        Examples:
            Simple spin hilbert space.

            >>> from netket.graph import Hypercube
            >>> from netket.hilbert import Qubit
            >>> g = Hypercube(length=10,n_dim=2,pbc=True)
            >>> hi = Qubit(graph=g)
            >>> print(hi.size)
            100
        """

        if(graph is None):
            graph = Edgeless(size)
        super().__init__(graph, [0, 1])
