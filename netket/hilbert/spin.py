from .custom_hilbert import CustomHilbert

import numpy as _np
from netket import random as _random
from numba import jit


class Spin(CustomHilbert):
    r"""Hilbert space obtained as tensor product of local spin states."""

    def __init__(self, graph, s, total_sz=None):
        r"""Hilbert space obtained as tensor product of local spin states.

        Args:
           graph: Graph representation of sites.
           s: Spin at each site. Must be integer or half-integer.
           total_sz: If given, constrains the total spin of system to a particular value.

        Examples:
           Simple spin hilbert space.

           >>> from netket.graph import Hypercube
           >>> from netket.hilbert import Spin
           >>> g = Hypercube(length=10,n_dim=2,pbc=True)
           >>> hi = Spin(graph=g, s=0.5)
           >>> print(hi.size)
           100
           """

        local_size = round(2 * s + 1)
        local_states = _np.empty(local_size)

        assert int(2 * s + 1) == local_size

        for i in range(local_size):
            local_states[i] = -round(2 * s) + 2 * i
        local_states = local_states.tolist()

        self._check_total_sz(total_sz, graph.n_nodes)
        if total_sz is not None:

            def constraints(x):
                return self._sum_constraint(x, total_sz)

        else:
            constraints = None

        self._total_sz = total_sz if total_sz is None else int(total_sz)
        self._s = s
        self._local_size = local_size

        super().__init__(graph, local_states, constraints)

    def random_state(self, out=None, rgen=None):
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
           >>> hi.random_state(rstate)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True
           """

        if out is None:
            out = _np.empty(self._size)

        if rgen is None:
            rgen = _random

        if self._total_sz is None:
            for i in range(self._size):
                rs = rgen.randint(0, self._local_size)
                out[i] = self.local_states[rs]
        else:
            sites = list(range(self.size))

            out.fill(-round(2 * self._s))
            ss = self.size

            for i in range(round(self._s * self.size) + self._total_sz):
                s = rgen.randint(0, ss)

                out[sites[s]] += 2

                if out[sites[s]] > round(2 * self._s - 1):
                    sites.pop(s)
                    ss -= 1

        return out

    @staticmethod
    @jit(nopython=True)
    def _sum_constraint(x, total_sz):
        return _np.sum(x, axis=1) == round(2 * total_sz)

    def _check_total_sz(self, total_sz, size):
        if total_sz is None:
            return

        m = round(2 * total_sz)
        if _np.abs(m) > size:
            raise Exception(
                "Cannot fix the total magnetization: 2|M| cannot " "exceed Nspins."
            )

        if (size + m) % 2 != 0:
            raise Exception(
                "Cannot fix the total magnetization: Nspins + " "totalSz must be even."
            )
