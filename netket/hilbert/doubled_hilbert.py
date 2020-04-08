from .abstract_hilbert import AbstractHilbert
from ..graph import DoubledGraph as _DoubledGraph

import numpy as _np
from netket import random as _random


class PyDoubledHilbert(AbstractHilbert):
    r"""Superoperatorial hilbert space for states living in the
    tensorised state H\otimes H, encoded according to Choi's isomorphism."""

    def __init__(self, hilb):
        """Superoperatorial hilbert space for states living in the
           tensorised state H\otimes H, encoded according to Choi's isomorphism.

        Args:
            hilb: the hilbrt space H.

        Examples:
            Simple superoperatorial hilbert space for few spins.

           >>> from netket.graph import Hypercube
           >>> from netket.hilbert import Spin, DoubledHilbert
           >>> g = Hypercube(length=5,n_dim=2,pbc=True)
           >>> hi = Spin(graph=g, s=0.5)
           >>> hi2 = DoubledHilbert(hi)
           >>> print(hi2.size)
           50
        """
        doubled_graph = _DoubledGraph(hilb.graph)

        self.graph = doubled_graph
        self._size = doubled_graph.size
        self.physical = hilb
        self._hilbert_index = None

        super().__init__()

    @property
    def size(self):
        return self._size

    @property
    def is_discrete(self):
        return self.physical.is_discrete

    @property
    def is_finite(self):
        return self.physical.is_finite

    @property
    def local_size(self):
        return self.physical.local_size

    @property
    def local_states(self):
        return self.physical.local_states

    @property
    def size_physical(self):
        return self.physical.size

    @property
    def n_states(self):
        return self.physical.n_states ** 2

    def numbers_to_states(self, numbers, out=None):
        if out is None:
            out = _np.empty((numbers.shape[0], self._size))

        # !!! WARNING
        # This code assumes that states are stored in a MSB
        # (Most Significant Bit) format.
        # We assume that the rightmost-half indexes the LSBs
        # and the leftmost-half indexes the MSBs
        # HilbertIndex-generated states respect this, as they are:
        # 0 -> [0,0,0,0]
        # 1 -> [0,0,0,1]
        # 2 -> [0,0,1,0]
        # etc...

        n = self.physical.size
        dim = self.physical.n_states
        left, right = _np.divmod(numbers, dim)

        self.physical.numbers_to_states(left, out=out[:, 0:n])
        self.physical.numbers_to_states(right, out=out[:, n : 2 * n])

        return out

    def states_to_numbers(self, states, out=None):
        if out is None:
            out = _np.empty(states.shape[0], _np.int64)

        # !!! WARNING
        # See note above in numbers_to_states

        n = self.physical.size
        dim = self.physical.n_states

        self.physical.states_to_numbers(states[:, 0:n], out=out)
        _out_l = out * dim

        self.physical.states_to_numbers(states[:, n : 2 * n], out=out)
        out += _out_l

        return out

    def random_vals(self, out=None, rgen=None):
        if out is None:
            out = _np.empty(self._size)

        if rgen is None:
            rgen = _random

        n = self.size_physical

        self.physical.random_vals(out=out[0:n], rgen=rgen)
        self.physical.random_vals(out=out[n : 2 * n], rgen=rgen)

        return out
