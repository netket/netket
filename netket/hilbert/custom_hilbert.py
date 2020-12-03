from .abstract_hilbert import AbstractHilbert
from .hilbert_index import HilbertIndex

from numba import jit
import numpy as _np
from netket import random as _random

from typing import Optional, List


class CustomHilbert(AbstractHilbert):
    r"""A custom hilbert space with discrete local quantum numbers."""

    def __init__(
        self,
        local_states: Optional[List[float]],
        N: int = 1,
        constraints: Optional = None,
    ):
        r"""
        Constructs a new ``CustomHilbert`` given a list of eigenvalues of the states and
        a number of sites, or modes, within this hilbert space.

        Args:
            local_states (list or None): Eigenvalues of the states. If the allowed states are an
                         infinite number, None should be passed as an argument.
            N: Number of modes in this hilbert space (default 1).
            constraints: A function specifying constraints on the quantum numbers.
                        Given a batch of quantum numbers it should return a vector
                        of bools specifying whether those states are valid or not.

        Examples:
           Simple custom hilbert space.

           >>> from netket.hilbert import CustomHilbert
           >>> g = Hypercube(length=10,n_dim=2,pbc=True)
           >>> hi = CustomHilbert(local_states=[-1232, 132, 0], N=100)
           >>> print(hi.size)
           100
        """

        assert isinstance(N, int)

        self._size = N

        self._is_finite = local_states is not None

        if self._is_finite:
            self._local_states = _np.asarray(local_states)
            assert self._local_states.ndim == 1
            self._local_size = self._local_states.shape[0]
            self._local_states = self._local_states.tolist()
        else:
            self._local_states = None
            self._local_size = _np.iinfo(_np.intp).max

        self._constraints = constraints
        self._do_constraints = self._constraints is not None

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
        r"""list[float] or None: A list of discreet local quantum numbers.
        If the local states are infinitely many, None is returned."""
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

    @property
    def is_finite(self):
        r"""bool: Whether the local hilbert space is finite."""
        return self._is_finite

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

        out = self._to_constrained_numbers_kernel(
            self._do_constraints,
            self._bare_numbers,
            hind.states_to_numbers(states, out),
        )

        return out

    def random_state(self, *, out=None, rgen=None):
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
           >>> hi = nk.hilbert.Boson(n_max=3, N=4)
           >>> rstate = hi.random_state()
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True
        """

        # Default version for discrete hilbert spaces without constraints
        # More specialized initializations can be defined in the derived classes
        if self.is_discrete and self.is_finite and not self._do_constraints:
            if out is None:
                out = _np.empty(self._size)

            if rgen is None:
                rgen = _random

            for i in range(self._size):
                rs = rgen.randint(0, self._local_size)
                out[i] = self.local_states[rs]

            return out
        
        return NotImplementedError

    def _get_hilbert_index(self):
        if self._hilbert_index is None:
            if not self.is_indexable:
                raise RuntimeError("The hilbert space is too large to be indexed.")

            self._hilbert_index = HilbertIndex(
                _np.asarray(self.local_states, dtype=_np.float64), self.size
            )

            if self._do_constraints:
                self._bare_numbers = self._gen_to_bare_numbers(
                    self._constraints(self._hilbert_index.all_states())
                )
            else:
                self._bare_numbers = _np.empty(0, dtype=_np.intp)

        return self._hilbert_index

    def _to_bare_numbers(self, numbers):
        if self._constraints is None:
            return numbers
        else:
            return self._bare_numbers[numbers]

    @staticmethod
    @jit(nopython=True)
    def _gen_to_bare_numbers(conditions):
        return _np.nonzero(conditions)[0]

    @staticmethod
    @jit(nopython=True)
    def _to_constrained_numbers_kernel(do_constraints, bare_numbers, numbers):
        if not do_constraints:
            return numbers
        else:
            found = _np.searchsorted(bare_numbers, numbers)
            if _np.max(found) >= bare_numbers.shape[0]:
                raise RuntimeError(
                    "The required state does not satisfy the given constraints."
                )
            return found

    def __pow__(self, n):
        if self._constraints is not None:
            raise NotImplementedError(
                """Cannot exponentiate a CustomHilbert with constraints. 
                Construct it from scratch instead."""
            )

        return CustomHilbert(self._local_states, self.size * n)

    def __repr__(self):
        constr = (
            ", #contraints={}".format(len(self._constraints))
            if self._do_constraints
            else ""
        )
        return "CustomHilbert(local_size={}{}; N={})".format(
            len(self.local_states), constr, self.size
        )
