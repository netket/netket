from typing import Optional

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

        super().__init__([0, 1], N)

    def _random_state_batch_impl(hilb, key, batches, dtype):
        rs = jax.random.randint(key, shape=(batches, hilb.size), minval=0, maxval=2)
        return jnp.asarray(rs, dtype=dtype)

    def __pow__(self, n):
        return Qubit(self.size * n)

    def __repr__(self):
        return "Qubit(N={})".format(self._size)
