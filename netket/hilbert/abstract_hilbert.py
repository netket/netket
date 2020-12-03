import abc
import numpy as _np

from typing import List, Tuple, Optional, Generator


"""int: Maximum number of states that can be indexed"""
max_states = _np.iinfo(_np.int32).max


class AbstractHilbert(abc.ABC):
    """Abstract class for NetKet hilbert objects"""

    @property
    @abc.abstractmethod
    def size(self):
        r"""int: The total number number of spins."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def is_discrete(self):
        r"""bool: Whether the hilbert space is discrete."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def is_finite(self):
        r"""bool: Whether the local hilbert space is finite."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def local_size(self):
        r"""int: Size of the local degrees of freedom that make the total hilbert space."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def local_states(self):
        r"""list[float]: A list of discreet local quantum numbers."""
        return NotImplementedError

    def numbers_to_states(self, numbers, out=None):
        r"""Returns the quantum numbers corresponding to the n-th basis state
        for input n. n is an array of integer indices such that numbers[k]=Index(states[k]).
        Throws an exception iff the space is not indexable.
        Args:
            numbers (numpy.array): Batch of input numbers to be converted into arrays of quantum numbers.
            out: Array of quantum numbers corresponding to numbers.
                 If None, memory is allocated.
        """
        return NotImplementedError

    def number_to_state(self, number):
        r"""Returns the quantum number corresponding to the n-th basis state
        for input n. n is a integer index such that number=Index(state).
        Throws an exception iff the space is not indexable.
        For a batch of numbers, prefer ```numbers_to_states```.

        Args:
            numbers (int or numpy.array): Input numbers to be converted into arrays
                                          of quantum numbers.

        Returns:
            int or numpy.array: A single number or an array (batched version) of
                                quantum numbers corresponding to the state.
        """
        if _np.isscalar(number):
            return self.numbers_to_states(_np.atleast_1d(number))[0, :]
        else:
            return self.numbers_to_states(number)

    def states_to_numbers(self, states, out=None):
        r"""Returns the basis state number corresponding to given quantum states.
        The states are given in a batch, such that states[k] has shape (hilbert.size).
        Throws an exception iff the space is not indexable.

        Args:
            states: Batch of states to be converted into the corresponding integers.
            out: Array of integers such that out[k]=Index(states[k]).
                 If None, memory is allocated.

        Returns:
            numpy.darray: Array of integers corresponding to out.
        """
        return NotImplementedError

    def state_to_number(self, state):
        r"""Returns the basis state number corresponding to given quantum states.
        Throws an exception iff the space is not indexable.
        For a batch of states, prefer ```states_to_numbers```.

        Args:
            state: A state or a batch of states to be converted into the corresponding integer.

        Returns:
            int: The index of the given input state.
        """
        if state.ndim == 1:
            return self.states_to_numbers(_np.atleast_2d(state))[0]
        elif state.ndim == 2:
            return self.states_to_numbers(state)
        else:
            raise RuntimeError("Invalid shape for state.")

    @property
    def n_states(self):
        r"""int: The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""
        return NotImplementedError

    def states(self):
        r"""Returns an iterator over all valid configurations of the Hilbert space.
        Throws an exception iff the space is not indexable.
        Iterating over all states with this method is typically inefficient,
        and ```all_states``` should be prefered.

        """
        for i in range(self.n_states):
            yield self.number_to_state(i).reshape(-1)

    @abc.abstractmethod
    def random_state(self, out=None, rgen=None):
        r"""Member function generating uniformely distributed local random states.

        Args:
            out: If provided, the random quantum numbers will be inserted into this array.
                 It should be of the appropriate shape and dtype.
            rgen: The random number generator. If None, the global
                  NetKet random number generator is used.
        """
        raise NotImplementedError

    def all_states(self, out=None):
        r"""Returns all valid states of the Hilbert space.
        Throws an exception iff the space is not indexable.
        Args:
            batch_size: If 'all' or None, all valid states in the Hilbert space are returned,
                        otherwise an iterator yielding batch_size states at the time.
                        If batch_size is not an integer multiple of the total number of states,
                        an error is returned.
            out: Optionally pre-allocated output.
        """
        numbers = _np.arange(0, self.n_states, dtype=_np.int64)
        return self.numbers_to_states(numbers, out)

    @property
    def is_indexable(self):
        if not self.is_discrete:
            return False

        if not self.is_finite:
            return False

        log_max = _np.log(max_states)

        return self.size * _np.log(self.local_size) <= log_max
