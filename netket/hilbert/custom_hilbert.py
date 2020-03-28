from .abstract_hilbert import AbstractHilbert
from .hilbert_index import HilbertIndex

from numba import jit
import numpy as _np


class PyCustomHilbert(AbstractHilbert):
    r"""A custom hilbert space with discrete local quantum numbers."""

    def __init__(self, graph, local_states, constraints=None):
        r"""
        Constructs a new ``CustomHilbert`` given a graph and a list of
        eigenvalues of the states.

        Args:
           graph: Graph representation of sites.
           local_states: Eigenvalues of the states.
           constraints: A function specifying constraints on the quantum numbers.
                        Given a batch of quantum numbers it should return a vector
                        of bools specifying whether those states are valid or not.

        Examples:
           Simple custom hilbert space.

           >>> from netket.graph import Hypercube
           >>> from netket.hilbert import CustomHilbert
           >>> g = Hypercube(length=10,n_dim=2,pbc=True)
           >>> hi = CustomHilbert(graph=g, local_states=[-1232, 132, 0])
           >>> print(hi.size)
           100
        """

        self.graph = graph
        self._size = graph.size

        self._local_states = _np.asarray(local_states)
        assert(self._local_states.ndim == 1)
        self._local_size = self._local_states.shape[0]
        self._local_states = self._local_states.tolist()
        self._constraints = constraints

        self._hilbert_index = None

        super().__init__()

    @property
    def size(self):
        r"""int: The total number number of degrees of freedom."""
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

    @property
    def n_states(self):
        r"""int: The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""

        hind = self._get_hilbert_index()

        if self._constraints is None:
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

            self._do_constraints = self._constraints is not None

            if(self._do_constraints):
                self._bare_numbers = self._gen_to_bare_numbers(
                    self._constraints(self._hilbert_index.all_states()))
            else:
                self._bare_numbers = None

        return self._hilbert_index

    def _to_bare_numbers(self, numbers):
        if self._constraints is None:
            return numbers
        else:
            return self._bare_numbers[numbers]

    @staticmethod
    @jit(nopython=True)
    def _gen_to_bare_numbers(conditions):
        return _np.argwhere(conditions).reshape(-1)

    def _to_constrained_numbers(self, numbers):
        return self._to_constrained_numbers_kernel(
            self._do_constraints, self._bare_numbers, numbers)

    @staticmethod
    @jit(nopython=True)
    def _to_constrained_numbers_kernel(do_constraints, bare_numbers, numbers):
        if do_constraints:
            return numbers

        found = _np.searchsorted(bare_numbers, numbers)
        if(_np.max(found) >= bare_numbers.shape[0]):
            raise RuntimeError(
                "The required state does not satisfy the given constraints.")
        return found
