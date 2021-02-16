from fractions import Fraction
from typing import Optional, List

import jax
from jax import numpy as jnp
import numpy as np
from netket.graph import AbstractGraph
from numba import jit

from .abstract_hilbert import AbstractHilbert
from .hilbert_index import HilbertIndex

import numpy as _np
import netket as nk


class TensorHilbert(AbstractHilbert):
    r"""Superoperatorial hilbert space for states living in the
    tensorised state H\otimes H, encoded according to Choi's isomorphism."""

    def __init__(self, *hilb_spaces):
        r"""Superoperatorial hilbert space for states living in the
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
        # if hasattr(hilb_1, "graph") and hasattr(hilb_2, "graph"):
        #    joint_graph = nk.graph.disjoint_union(hilb_1.graph, hilb_2.graph)
        #    self.graph = joint_graph

        self._hilbert_spaces = hilb_spaces
        self._n_hilbert_spaces = len(hilb_spaces)
        self._hilbert_i = _np.concatenate(
            [[i for _ in range(hi.size)] for (i, hi) in enumerate(hilb_spaces)]
        )

        self._sizes = tuple([hi.size for hi in hilb_spaces])
        self._cum_sizes = _np.cumsum(self._sizes)
        self._cum_indices = _np.concatenate([[0], self._cum_sizes])
        self._size = sum(self._sizes)

        self._ns_states = [hi.n_states for hi in self._hilbert_spaces]
        self._ns_states_r = _np.flip(self._ns_states)
        self._cum_ns_states = _np.concatenate([[0], _np.cumprod(self._ns_states)])
        self._cum_ns_states_r = _np.concatenate(
            [_np.flip(_np.cumprod(self._ns_states)[:-1]), [1]]
        )
        self._n_states = np.prod(self._ns_states)

        self._delta_indices_i = _np.array(
            [self._cum_indices[i] for i in self._hilbert_i]
        )

        super().__init__()

    @property
    def size(self) -> int:
        return self._size

    @property
    def shape(self) -> Tuple[int]:
        return self._sizes

    @property
    def is_discrete(self):
        return all([hi.is_discrete for hi in self._hilbert_spaces])

    @property
    def is_finite(self):
        return all([hi.is_finite for hi in self._hilbert_spaces])

    def _sub_index(self, i):
        for (j, sz) in enumerate(self._cum_sizes):
            if i < sz:
                return j

    def size_at_index(self, i):
        # j = self._sub_index(i)
        # return self._hilbert_spaces[j].size_at_index(i-self._cum_indices[j-1])
        return self._hilbert_spaces[self._hilbert_i[i]].size_at_index(
            i - self._delta_indices_i[i]
        )

    def states_at_index(self, i):
        # j = self._sub_index(i)
        # return self._hilbert_spaces[j].states_at_index(i-self._cum_indices[j-1])
        return self._hilbert_spaces[self._hilbert_i[i]].states_at_index(
            i - self._delta_indices_i[i]
        )

    @property
    def n_states(self):
        return self._n_states

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

        rem = numbers
        for (i, dim) in enumerate(self._ns_states_r):
            rem, loc_numbers = _np.divmod(rem, dim)
            hi_i = self._n_hilbert_spaces - (i + 1)
            self._hilbert_spaces[hi_i].numbers_to_states(
                loc_numbers, out=out[:, self._cum_indices[hi_i] : self._cum_sizes[hi_i]]
            )

        return out

    def states_to_numbers(self, states, out=None):
        if out is None:
            out = _np.zeros(states.shape[0], _np.int64)

        temp = out.copy()

        # !!! WARNING
        # See note above in numbers_to_states

        for (i, dim) in enumerate(self._cum_ns_states_r):
            self._hilbert_spaces[i].states_to_numbers(
                states[:, self._cum_indices[i] : self._cum_sizes[i]], out=temp
            )
            out += temp * dim

        return out

    def __repr__(self):
        _str = "{}".format(self._hilbert_spaces[0])
        for hi in self._hilbert_spaces[1:]:
            _str += "*{}".format(hi)

        return _str

    @property
    def _attrs(self):
        return self._hilbert_spaces

    def __mul__(self, other):
        return TensorHilbert(*self._hilbert_spaces, other)
