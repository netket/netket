from typing import Optional, Union, Iterable, List

import jax
from jax import numpy as jnp

from netket.graph import AbstractGraph

from .custom_hilbert import CustomHilbert
from ._deprecations import graph_to_N_depwarn


class Qubit(CustomHilbert):
    r"""Hilbert space obtained as tensor product of local qubit states."""

    def __init__(self, N: int = 1, graph: Optional[AbstractGraph] = None):
        r"""Initializes a qubit hilbert space.

        Args:
        N: Number of qubits.
        graph: (deprecated) a graph from which to extract the number of sites.

        Examples:
            Simple spin hilbert space.

            >>> from netket.graph import Hypercube
            >>> from netket.hilbert import Qubit
            >>> g = Hypercube(length=10,n_dim=2,pbc=True)
            >>> hi = Qubit(graph=g)
            >>> print(hi.size)
            100
        """
        N = graph_to_N_depwarn(N=N, graph=graph)

        super().__init__([0.0, 1.0], N)

    def __pow__(self, n):
        return Qubit(self.size * n)

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        return Qubit(self.size + other.size)

    def ptrace(self, sites: Union[int, List]) -> Optional["Qubit"]:
        if isinstance(sites, int):
            sites = [sites]

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            return Qubit(N=self.size - Nsites)

    def __repr__(self):
        return "Qubit(N={})".format(self._size)

    @property
    def _attrs(self):
        return (self.size,)
