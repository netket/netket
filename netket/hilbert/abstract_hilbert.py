import abc
from functools import partial

from typing import List, Tuple, Optional, Generator

import jax
import numpy as np

from netket.utils import deprecated


"""int: Maximum number of states that can be indexed"""
max_states = np.iinfo(np.int32).max


class AbstractHilbert(abc.ABC):
    """Abstract class for NetKet hilbert objects"""

    @property
    @abc.abstractmethod
    def size(self):
        r"""int: The total number number of spins."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_discrete(self):
        r"""bool: Whether the hilbert space is discrete."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_finite(self):
        r"""bool: Whether the local hilbert space is finite."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def local_size(self):
        r"""int: Size of the local degrees of freedom that make the total hilbert space."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def local_states(self):
        r"""list[float]: A list of discreet local quantum numbers."""
        raise NotImplementedError()

    def numbers_to_states(self, numbers, out=None):
        r"""Returns the quantum numbers corresponding to the n-th basis state
        for input n. n is an array of integer indices such that numbers[k]=Index(states[k]).
        Throws an exception iff the space is not indexable.
        Args:
            numbers (numpy.array): Batch of input numbers to be converted into arrays of quantum numbers.
            out: Array of quantum numbers corresponding to numbers.
                 If None, memory is allocated.
        """
        raise NotImplementedError()

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
        if np.isscalar(number):
            return self.numbers_to_states(np.atleast_1d(number))[0, :]
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
        raise NotImplementedError()

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
            return self.states_to_numbers(np.atleast_2d(state))[0]
        elif state.ndim == 2:
            return self.states_to_numbers(state)
        else:
            raise RuntimeError("Invalid shape for state.")

    @property
    def n_states(self):
        r"""int: The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""
        raise NotImplementedError()

    def states(self):
        r"""Returns an iterator over all valid configurations of the Hilbert space.
        Throws an exception iff the space is not indexable.
        Iterating over all states with this method is typically inefficient,
        and ```all_states``` should be prefered.

        """
        for i in range(self.n_states):
            yield self.number_to_state(i).reshape(-1)

    def random_state(self, key, size=None, dtype=np.float32):
        r"""Generates either a single or a batch of uniformly distributed random states.

        Args:
            size: If provided, returns a batch of configurations of the form (size, #) if size
                is an integer or (*size, #) if it is a tuple and where # is the Hilbert space size.
                By default, a single random configuration with shape (#,) is returned.
            out: If provided, the random quantum numbers will be inserted into this array,
                 which should be of the appropriate shape (see `size`) and data type.
            rgen: The random number generator. If None, the global NetKet random
                number generator is used.

        Example:
            >>> hi = netket.hilbert.Qubit(N=2)
            >>> hi.random_state()
            array([0., 1.])
            >>> hi.random_state(size=2)
            array([[0., 0.], [1., 0.]])
        """
        from netket.hilbert import random

        return random.random_state(self, key, size, dtype=dtype)

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
        numbers = np.arange(0, self.n_states, dtype=np.int64)
        return self.numbers_to_states(numbers, out)

    @property
    def is_indexable(self):
        if not self.is_discrete:
            return False

        if not self.is_finite:
            return False

        log_max = np.log(max_states)

        return self.size * np.log(self.local_size) <= log_max

    @partial(jax.jit, static_argnums=(0, 2))
    def _random_state_scalar(hilb, key, dtype):
        """
        Generates a single random state-vector given an hilbert space and a rng key.
        """
        # Attempt to use the scalar method
        res = hilb._random_state_scalar_impl(key, dtype)
        if res is NotImplemented:
            # If the scalar method is not implemented, use the batch method and take the first batch
            res = hilb._random_state_batch_impl(key, 1, dtype).reshape(-1)
            if res is NotImplemented:
                raise NotImplementedError(
                    """_jax_random_state_scalar(hilb, key, dtype) is not defined for hilb of type {} or a supertype.
                    For better performance you should define _jax_random_state_batch.""".format(
                        type(hilb)
                    )
                )

        return res

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _random_state_batch(hilb, key, size, dtype):
        """
        Generates a batch of random state-vectors given an hilbert space and a rng key.
        """
        # Attempt to use the batch method
        res = hilb._random_state_batch_impl(key, size, dtype)
        if res is NotImplemented:
            # If it fails, vmap over the scalar method
            keys = jax.random.split(key, size)
            res = jax.vmap(
                hilb._random_state_scalar_impl, in_axes=(None, 0, None), out_axes=0
            )(hilb, key, dtype)
        return res

    def _random_state_scalar_impl(hilb, key, dtype):
        # Implementation for jax_random_state_scalar, dispatching on the
        # type of hilbert.
        # Could probably put it in the class itself (which @singledispatch automatically
        # because of oop)?
        return NotImplemented

    def _random_state_batch_impl(hilb, key, size, dtype):
        # Implementation for jax_random_state_batch, dispatching on the
        # type of hilbert.
        # Could probably put it in the class itself (which @singledispatch automatically
        # because of oop)?
        return NotImplemented
