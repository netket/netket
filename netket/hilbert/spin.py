from .abstract_hilbert import AbstractHilbert

__all__ = ["Spin"]

import numpy as _np
from netket import random as _random


class PySpin(AbstractHilbert):

    def __init__(self, graph, s, total_sz=None):
        self._s = s
        self.graph = graph
        self._size = graph.size
        self._local_size = round(2 * s + 1)
        self._local_states = _np.empty(self._local_size)

        assert(int(2 * s + 1) == self._local_size)

        for i in range(self._local_size):
            self._local_states[i] = -round(2 * s) + 2 * i
        self._local_states = self._local_states.tolist()

        super().__init__()

    @property
    def size(self):
        r"""int: The total number number of spins."""
        return self._size

    @property
    def is_discrete(self):
        r"""bool: Whether the hilbert space is discrete."""
        return True

    @property
    def local_size(self):
        r"""int: Size of the local degrees of freedom that make the total hilbert space."""
        return self._local_size

    @property
    def local_states(self):
        r"""list[float]: A list of discreet local quantum numbers."""
        return self._local_states

    def random_vals(self, out=None, rgen=None):
        r"""Member function generating uniformely distributed local random states.

        Args:
            out: If provided, the random quantum numbers will be inserted into this array.
                 It should be of the appropriate shape and dtype.
            rgen: The random number generator. If None, the global
                  NetKet random number generator is used.

        Examples:
           Test that a new random state is a possible state for the hilbert
           space.

           >>> import netket as nk
           >>> import numpy as np
           >>> hi = nk.hilbert.Boson(n_max=3, graph=nk.graph.Hypercube(length=5, n_dim=1))
           >>> rstate = np.zeros(hi.size)
           >>> rg = nk.utils.RandomEngine(seed=1234)
           >>> hi.random_vals(rstate, rg)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True
           """

        if out is None:
            out = _np.empty(self._size)

        if(rgen is None):
            rgen = _random

        for i in range(self._size):
            rs = rgen.randint(0, self._local_size)
            out[i] = self.local_states[rs]

        return out
