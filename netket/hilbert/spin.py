from .abstract_hilbert import AbstractHilbert
from .hilbert_index import HilbertIndex

__all__ = ["Spin"]

import numpy as _np
from netket import random as _random
from numba import jit


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

        self._total_sz = total_sz

        self._check_total_sz()

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

        if(self._total_sz is None):
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

                if(out[sites[s]] > round(2 * self._s - 1)):
                    sites.pop(s)
                    ss -= 1

        return out

    def _check_total_sz(self):
        if(self._total_sz is None):
            return

        m = round(2 * self._total_sz)
        if (_np.abs(m) > self.size):
            raise Exception(
                "Cannot fix the total magnetization: 2|M| cannot "
                "exceed Nspins.")

        if ((self.size + m) % 2 != 0):
            raise Exception(
                "Cannot fix the total magnetization: Nspins + "
                "totalSz must be even.")

    @property
    def n_states(self):
        r"""int: The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""

        hind = self._get_hilbert_index()

        if self._total_sz is None:
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
        if(not self.is_indexable()):
            raise Exception('The hilbert space is too large to be indexed.')

        if(self._hilbert_index is None):
            self._hilbert_index = HilbertIndex(_np.asarray(
                self.local_states), self.local_size, self.size)

            if(self._total_sz is not None):
                self._bare_numbers = self._gen_to_bare_numbers(
                    self._total_sz, self._hilbert_index.all_states())
            else:
                self._bare_numbers = None

        return self._hilbert_index

    def _to_bare_numbers(self, numbers):
        if self._total_sz is None:
            return numbers
        else:
            return self._bare_numbers[numbers]

    @staticmethod
    @jit(nopython=True)
    def _gen_to_bare_numbers(total_sz, bare_states):
        conditions = (bare_states.sum(axis=1) == round(2 * total_sz))
        return _np.argwhere(conditions).reshape(-1)

    def _to_constrained_numbers(self, numbers):
        return self._to_constrained_numbers_kernel(
            self._total_sz, self._bare_numbers, numbers)

    @staticmethod
    @jit(nopython=True)
    def _to_constrained_numbers_kernel(total_sz, bare_numbers, numbers):
        if total_sz is None:
            return numbers

        found = _np.searchsorted(bare_numbers, numbers)
        if(_np.max(found) >= bare_numbers.shape[0]):
            raise RuntimeError(
                "The required state does not satisfy the total_sz constraint.")
        return found
