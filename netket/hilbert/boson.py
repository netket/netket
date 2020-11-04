from .custom_hilbert import CustomHilbert

import numpy as _np
from netket import random as _random
from numba import jit


class Boson(CustomHilbert):
    r"""Hilbert space obtained as tensor product of local bosonic states."""

    def __init__(self, graph, n_max=None, n_bosons=None):
        r"""
        Constructs a new ``Boson`` given a graph,  maximum occupation number,
        and total number of bosons.

        Args:
           graph: Graph representation of sites.
           n_max: Maximum occupation for a site (inclusive). If None, the local occupation
                  number is unbounded.
           n_bosons: Constraint for the number of bosons. If None, no constraint
                  is imposed.

        Examples:
           Simple boson hilbert space.

           >>> from netket.graph import Hypercube
           >>> from netket.hilbert import Boson
           >>> g = Hypercube(length=10,n_dim=2,pbc=True)
           >>> hi = Boson(graph=g, n_max=5, n_bosons=11)
           >>> print(hi.size)
           100
        """

        self._n_max = n_max

        if n_bosons is not None:
            n_bosons = int(n_bosons)
            self._n_bosons = n_bosons
            assert n_bosons > 0

            if self._n_max is None:
                self._n_max = n_bosons
            else:
                if self._n_max * graph.n_nodes < n_bosons:
                    raise Exception(
                        """The required total number of bosons is not compatible
                        with the given n_max."""
                    )

            def constraints(x):
                return self._sum_constraint(x, n_bosons)

        else:
            constraints = None
            self._n_bosons = None

        if self._n_max is not None:
            assert self._n_max > 0
            local_states = _np.arange(self._n_max + 1)
        else:
            max_ind = _np.iinfo(_np.intp).max
            self._n_max = max_ind
            local_states = None

        self._hilbert_index = None

        super().__init__(graph, local_states, constraints)

    @property
    def n_max(self):
        r"""int or None: The maximum number of bosons per site, or None
        if the number is unconstrained."""
        return self._n_max

    @property
    def n_bosons(self):
        r"""int or None: The total number of particles, or None
        if the number is unconstrained."""
        return self._n_bosons

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
           >>> rg = nk.utils.RandomEngine(seed=1234)
           >>> hi.random_vals(rstate, rg)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True
        """

        if out is None:
            out = _np.empty(self.size)

        if rgen is None:
            rgen = _random

        if self.n_bosons is None:
            for i in range(self.size):
                rs = rgen.randint(0, self.local_size)
                if self.is_finite:
                    out[i] = self.local_states[rs]
                else:
                    out[i] = rs
        else:
            sites = list(range(self.size))

            out.fill(0.0)
            ss = self.size

            for i in range(self.n_bosons):
                s = rgen.randint(0, ss)

                out[sites[s]] += 1

                if out[sites[s]] == self.n_max:
                    sites.pop(s)
                    ss -= 1

        return out

    @staticmethod
    @jit(nopython=True)
    def _sum_constraint(x, n_bosons):
        return _np.sum(x, axis=1) == n_bosons

    def __repr__(self):
        nbosons = (
            ", n_bosons={}".format(self._n_bosons) if self._n_bosons is not None else ""
        )
        nmax = self._n_max if self._n_max < _np.iinfo(_np.intp).max else "INT_MAX"
        return "Boson(n_max={}{}; N={})".format(nmax, nbosons, self._size)
