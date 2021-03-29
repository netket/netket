# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from functools import partial

from typing import List, Tuple, Optional, Generator, Union, Iterable, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import deprecated


"""int: Maximum number of states that can be indexed"""
max_states = np.iinfo(np.int32).max


class NoneType:
    pass


class AbstractHilbert(abc.ABC):
    """Abstract class for NetKet hilbert objects.

    This class definese the common interface that can be used to
    interact with hilbert spaces.

    Hilbert Spaces are immutable.
    """

    def __init__(self):
        self._hash = None

    @property
    @abc.abstractmethod
    def size(self) -> int:
        r"""The total number number of spins."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, ...]:
        r"""The size of the hilbert space on every site."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_discrete(self) -> bool:
        r"""Whether the hilbert space is discrete."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_finite(self) -> bool:
        r"""Whether the local hilbert space is finite."""
        raise NotImplementedError()

    @abc.abstractmethod
    def size_at_index(self, i: int) -> int:
        r"""Size of the local degrees of freedom at the site i.

        Args:
            i: The index of the desired site

        Returns:
            The number of degrees of freedom at that site
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def states_at_index(self, i: int) -> Optional[List[float]]:
        r"""A list of discrete local quantum numbers at the site i.
        If the local states are infinitely many, None is returned.

        Args:
            i: The index of the desired site.

        Returns:
            A list of values or None if there are infintely many.
        """
        raise NotImplementedError()

    def numbers_to_states(
        self, numbers: Union[int, np.ndarray], out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        r"""Returns the quantum numbers corresponding to the n-th basis state
        for input n. n is an array of integer indices such that numbers[k]=Index(states[k]).
        Throws an exception iff the space is not indexable.

        Args:
            numbers (numpy.array): Batch of input numbers to be converted into arrays of quantum numbers.
            out: Optional Array of quantum numbers corresponding to numbers.
        """
        if out is None:
            out = np.empty((np.atleast_1d(numbers).shape[0], self._size))

        if np.any(numbers >= self.n_states):
            raise ValueError("numbers outside the range of allowed states")

        if np.isscalar(numbers):
            return self._numbers_to_states(np.atleast_1d(numbers), out=out)[0, :]
        else:
            return self._numbers_to_states(numbers, out=out)

    def states_to_numbers(
        self, states: np.ndarray, out: Optional[np.ndarray] = None
    ) -> Union[int, np.ndarray]:
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
        if states.shape[-1] != self.size:
            raise ValueError(
                f"""Size of this state ({states.shape[-1]}) not 
                                 corresponding to this hilbert space {self.size}
                                 """
            )

        states_r = np.asarray(np.reshape(states, (-1, states.shape[-1])))

        if out is None:
            out = np.empty(states_r.shape[:-1], dtype=np.int64)

        out = self._states_to_numbers(states_r, out=out.reshape(-1))

        if states.ndim == 1:
            return out[0]
        else:
            return out.reshape(states.shape[:-1])

    @property
    def n_states(self) -> int:
        r"""The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""
        raise NotImplementedError()

    def states(self) -> Iterator[np.ndarray]:
        r"""Returns an iterator over all valid configurations of the Hilbert space.
        Throws an exception iff the space is not indexable.
        Iterating over all states with this method is typically inefficient,
        and ```all_states``` should be prefered.

        """
        for i in range(self.n_states):
            yield self.numbers_to_states(i).reshape(-1)

    # after removing legacy:
    # signature must be the following
    # def random_state(self, key, size=None, dtype=np.float32):
    def random_state(
        self,
        key=NoneType(),
        size: Optional[int] = NoneType(),
        dtype=np.float32,
        out: Optional[np.ndarray] = None,
        rgen=None,
    ) -> jnp.ndarray:
        r"""Generates either a single or a batch of uniformly distributed random states.
        Runs as :code:`random_state(self, key, size=None, dtype=np.float32)` by default.

        Args:
            key: rng state from a jax-style functional generator.
            size: If provided, returns a batch of configurations of the form :code:`(size, N)` if size
                  is an integer or :code:`(*size, N)` if it is a tuple and where :math:`N` is the Hilbert space size.
                  By default, a single random configuration with shape :code:`(#,)` is returned.
            dtype: DType of the resulting vector.
            out: Deprecated. Will be removed in v3.1
            rgen: Deprecated. Will be removed in v3.1

        Returns:
            A state or batch of states sampled from the uniform distribution on the hilbert space.

        Example:

            >>> hi = netket.hilbert.Qubit(N=2)
            >>> hi.random_state(jax.random.PRNGKey(0))
            array([0., 1.])
            >>> hi.random_state(size=2)
            array([[0., 0.], [1., 0.]])
        """
        # legacy support
        # TODO: Remove in 3.1
        # if no positional arguments, and key is unspecified -> legacy
        if isinstance(key, NoneType):
            # legacy sure
            if isinstance(size, NoneType):
                return self._random_state_legacy(size=None, out=out, rgen=rgen)
            else:
                return self._random_state_legacy(size=size, out=out, rgen=rgen)
        elif (
            isinstance(key, tuple)
            or isinstance(key, int)
            and isinstance(size, NoneType)
        ):
            # if one positional argument legacy typee...
            return self._random_state_legacy(size=key, out=out, rgen=rgen)
        else:
            from netket.hilbert import random

            size = size if not isinstance(size, NoneType) else None

            return random.random_state(self, key, size, dtype=dtype)

    def all_states(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        r"""Returns all valid states of the Hilbert space.

        Throws an exception if the space is not indexable.

        Args:
            out: an optional pre-allocated output array

        Returns:
            A (n_states x size) batch of statess. this corresponds
            to the pre-allocated array if it was passed.
        """
        numbers = np.arange(0, self.n_states, dtype=np.int64)

        return self.numbers_to_states(numbers, out)

    def ptrace(self, sites: Union[int, Iterable]) -> "AbstractHilbert":
        """Returns the hilbert space without the selected sites.

        Not all hilbert spaces support this operation.

        Args:
            sites: a site or list of sites to trace away

        Returns:
            The partially-traced hilbert space. The type of the resulting hilbert space
            might be different from the starting one.
        """
        raise NotImplementedError("Ptrace not implemented for this hilbert space type")

    @property
    def is_indexable(self) -> bool:
        """"Whever the space can be indexed with an integer"""
        if not self.is_discrete:
            return False

        if not self.is_finite:
            return False

        log_max = np.log(max_states)

        return np.sum(np.log(self.shape)) <= log_max

    def __mul__(self, other: "AbstractHilbert"):
        if self == other:
            return self ** 2
        else:
            from .tensor_hilbert import TensorHilbert

            if type(self) == type(other):
                res = self._mul_sametype_(other)
                if res is not NotImplemented:
                    return res

            return TensorHilbert(self) * other

    @property
    @abc.abstractmethod
    def _attrs(self) -> Tuple:
        """
        Tuple of hashable attributs, used to compute the immutable
        hash of this Hilbert space
        """
        pass

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return self._attrs == other._attrs

        return False

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._attrs)

        return self._hash
