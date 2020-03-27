from .abstract_hilbert import AbstractHilbert
from .hilbert_index import HilbertIndex

import numpy as _np
from netket import random as _random
from numba import jit


class PyBoson(AbstractHilbert):
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

        self.graph = graph
        self._size = graph.size
        self._n_max = n_max
        self._n_bosons = n_bosons

        if(n_bosons is not None):
            assert(n_bosons > 0)

            if(self._n_max is None):
                self._n_max = n_bosons
            else:
                if(self._n_max * graph.size < n_bosons):
                    raise Exception(
                        """The required total number of bosons is not compatible
                        with the given n_max.""")

        if(self._n_max is not None):
            assert(self._n_max > 0)
            self._local_size = self._n_max + 1
            self._local_states = _np.arange(self._n_max + 1).tolist()
        else:
            max_ind = _np.iinfo(_np.intp).max
            self._local_size = max_ind
            self._local_states = lambda x: x

        self._hilbert_index = None

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
        r"""list[float]: A list of discreet allowed local quantum
                         numbers if the local occupation number
                         is bounded. """
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

        if(self._n_bosons is None):
            for i in range(self._size):
                rs = rgen.randint(0, self._local_size)
                if(callable(self.local_states)):
                    out[i] = self.local_states(rs)
                else:
                    out[i] = self.local_states[rs]
        else:
            sites = list(range(self.size))

            out.fill(0.)
            ss = self.size

            for i in range(self._n_bosons):
                s = rgen.randint(0, ss)

                out[sites[s]] += 1

                if(out[sites[s]] > self._n_max):
                    sites.pop(s)
                    ss -= 1

        return out

    @property
    def n_states(self):
        r"""int: The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""

        hind = self._get_hilbert_index()

        if self._n_bosons is None:
            return hind.n_states
        else:
            return self._bare_numbers.shape[0]

    def numbers_to_states(self, numbers, out=None):
        r"""Returns the quantum numbers corresponding to the n-th basis state
        for input n. n is an array of integer indices such that numbers[k]=Index(states[k]).
        Throws an exception iff the space is not indexable.
        Args:
            numbers: Batch of input numbers to be converted into arrays of quantum numbers.
            out: Array of quantum numbers corresponding to numbers.
                 If None, memory is allocated.
        """

        hind = self._get_hilbert_index()
        return hind.numbers_to_states(self._to_bare_numbers(numbers), out)

    def states_to_numbers(self, states, out=None):
        r"""Returns the basis state number corresponding to given quantum states.
        The states are given in a batch, such that states[k] has shape (hilbert.size).
        Throws an exception iff the space is not indexable.
        Args:
            states: Batch of states to be converted into the corresponding integers.
            out: Array of integers such that out[k]=Index(states[k]).
                 If None, memory is allocated.
        """
        hind = self._get_hilbert_index()

        out = self._to_constrained_numbers(hind.states_to_numbers(states, out))

        return out

    def _get_hilbert_index(self):
        if(self._hilbert_index is None):
            if(not self.is_indexable()):
                raise Exception(
                    'The hilbert space is too large to be indexed.')

            self._hilbert_index = HilbertIndex(_np.asarray(
                self.local_states, dtype=_np.float64), self.local_size, self.size)

            if(self._n_bosons is not None):
                self._bare_numbers = self._gen_to_bare_numbers(
                    self._n_bosons, self._hilbert_index.all_states())
            else:
                self._bare_numbers = None

        return self._hilbert_index

    def _to_bare_numbers(self, numbers):
        if self._n_bosons is None:
            return numbers
        else:
            return self._bare_numbers[numbers]

    @staticmethod
    @jit(nopython=True)
    def _gen_to_bare_numbers(n_bosons, bare_states):
        conditions = (bare_states.sum(axis=1) == n_bosons)
        return _np.argwhere(conditions).reshape(-1)

    def _to_constrained_numbers(self, numbers):
        return self._to_constrained_numbers_kernel(
            self._n_bosons, self._bare_numbers, numbers)

    @staticmethod
    @jit(nopython=True)
    def _to_constrained_numbers_kernel(n_bosons, bare_numbers, numbers):
        if n_bosons is None:
            return numbers

        found = _np.searchsorted(bare_numbers, numbers)
        if(_np.max(found) >= bare_numbers.shape[0]):
            raise RuntimeError(
                "The required state does not satisfy the total_sz constraint.")
        return found
