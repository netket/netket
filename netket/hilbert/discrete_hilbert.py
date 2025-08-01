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

from collections.abc import Iterator
from textwrap import dedent
from functools import partial, reduce

import numpy as np

import jax
import jax.numpy as jnp

from equinox import error_if

from netket.utils.types import Array, DType
from netket.jax import sharding

from .abstract_hilbert import AbstractHilbert
from .index import is_indexable


class DiscreteHilbert(AbstractHilbert):
    """Abstract class for an hilbert space defined on a lattice.

    This class defines the common interface that can be used to
    interact with hilbert spaces on lattices.

    The local degrees of freedom are discrete and numerable, therefore they can always be
    converted to and from integers using the methods `states_to_local_indices` and
    `local_indices_to_states`.
    This can be used to simplify the implementation of operators that might act on the
    hilbert space, to avoid reimplementing the logic for different values of the local
    degrees of freedom.

    If the Hilbert space is small enough, individual states can be converted to and from
    integers labelling all the basis states. This is done using the methods `numbers_to_states`
    and `states_to_numbers`.
    """

    def __init__(self, shape: tuple[int, ...]):
        """
        Initializes a discrete Hilbert space with a basis of given shape.

        Args:
            shape: The local dimension of the Hilbert space for each degree
                of freedom.
        """
        self._shape = tuple(shape)

        super().__init__()

    @property
    def shape(self) -> tuple[int, ...]:
        r"""The size of the hilbert space on every site."""
        return self._shape

    @property
    def constrained(self) -> bool:
        r"""The hilbert space does not contains `prod(hilbert.shape)`
        basis states.

        Typical constraints are population constraints (such as fixed
        number of bosons, fixed magnetization...) which ensure that
        only a subset of the total unconstrained space is populated.

        Typically, objects defined in the constrained space cannot be
        converted to QuTiP or other formats.
        """
        raise NotImplementedError(  # pragma: no cover
            dedent(
                f"""
            `constrained` is not implemented for discrete hilbert
            space {type(self)}.
            """
            )
        )

    @property
    def is_finite(self) -> bool:
        r"""Whether the local hilbert space is finite."""
        raise NotImplementedError(  # pragma: no cover
            dedent(
                f"""
            `is_finite` is not implemented for discrete hilbert
            space {type(self)}.
            """
            )
        )

    @property
    def n_states(self) -> int:
        r"""The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""
        raise NotImplementedError(  # pragma: no cover
            dedent(
                f"""
            `n_states` is not implemented for discrete hilbert
            space {type(self)}.
            """
            )
        )

    def size_at_index(self, i: int) -> int:
        r"""Size of the local degrees of freedom for the i-th variable.

        Args:
            i: The index of the desired site

        Returns:
            The number of degrees of freedom at that site
        """
        return self.shape[i]  # pragma: no cover

    def states_at_index(self, i: int) -> list[float] | None:
        r"""A list of discrete local quantum numbers at the site i.

        If the local states are infinitely many, None is returned.

        Args:
            i: The index of the desired site.

        Returns:
            A list of values or None if there are infinitely many.
        """
        raise NotImplementedError()  # pragma: no cover

    @partial(jax.jit, static_argnums=0)
    def numbers_to_states(self, numbers: Array) -> jax.Array:
        r"""Returns the quantum numbers corresponding to the n-th basis state
        for input n.

        `n` is an array of integer indices such that
        :code:`numbers[k]=Index(states[k])`.
        Throws an exception iff the space is not indexable.

        This function validates the inputs by checking that the numbers provided
        are smaller than the Hilbert space size, and throws an error if that
        condition is not met. When called from within a `jax.jit` context, this
        uses {func}`equinox.error_if` to throw runtime errors.

        Args:
            numbers (numpy.array): Batch of input numbers to be converted into arrays of
                quantum numbers.
        """

        if not self.is_indexable:
            raise RuntimeError("The hilbert space is too large to be indexed.")

        numbers = jnp.asarray(numbers, dtype=np.int32)

        # equinox.error_if is broken under shard_map.
        # If we are using shard map, we skip this check
        if sharding._get_SHARD_MAP_STACK_LEVEL() == 0 and jax.device_count() == 1:
            numbers = error_if(
                numbers,
                (numbers >= self.n_states).any() | (numbers < 0).any(),
                "Numbers outside the range of allowed states.",
            )

        return self._numbers_to_states(numbers.ravel()).reshape(
            (*numbers.shape, self.size)
        )

    def _numbers_to_states(self, numbers: jax.Array) -> jax.Array:
        """
        This method must be overriden by subclasses to allow conversion of
        numbers to states.

        Args:
            numbers: jax array encoding a batch of input numbers to be converted
                into arrays of quantum numbers.

        Returns:
            A batch of states.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=0)
    def states_to_numbers(self, states: Array) -> jax.Array:
        r"""Returns the basis state number corresponding to given quantum states.

        The states are given in a batch, such that states[k] has shape (hilbert.size).
        Throws an exception iff the space is not indexable.

        Args:
            states: Batch of states to be converted into the corresponding integers.

        Returns:
            numpy.darray: Array of integers corresponding to states.
        """

        if states.shape[-1] != self.size:
            raise ValueError(
                f"Size of this state ({states.shape[-1]}) not"
                f"corresponding to this hilbert space {self.size}"
            )

        states_r = jnp.reshape(states, (-1, states.shape[-1]))

        if not self.is_indexable:
            raise RuntimeError("The hilbert space is too large to be indexed.")

        out = self._states_to_numbers(states_r)

        if states.ndim == 1:
            return out[0]
        else:
            return out.reshape(states.shape[:-1])

    def _states_to_numbers(self, states: jax.Array) -> jax.Array:
        """
        This method must be overriden by subclasses to allow conversion of
        states to numbers.

        Args:
            states: jax array encoding a batch of input states to be converted
                into a vector of numbers.

        Returns:
            A vector of numbers.
        """
        raise NotImplementedError

    def states(self) -> Iterator[np.ndarray]:
        r"""Returns an iterator over all valid configurations of the Hilbert space.
        Throws an exception iff the space is not indexable.
        Iterating over all states with this method is typically inefficient,
        and ```all_states``` should be preferred.

        """
        for i in range(self.n_states):
            yield self.numbers_to_states(i).reshape(-1)

    @partial(jax.jit, static_argnums=0)
    def all_states(self) -> Array:
        r"""Returns all valid states of the Hilbert space.

        Throws an exception if the space is not indexable.

        Returns:
            A (n_states x size) batch of states. this corresponds
            to the pre-allocated array if it was passed.
        """

        numbers = jnp.arange(0, self.n_states, dtype=np.int32)
        return self.numbers_to_states(numbers)

    def states_to_local_indices(self, x: Array):
        r"""Returns a tensor with the same shape of `x`, where all local
        values are converted to indices in the range `0...self.shape[i]`.
        This function is guaranteed to be jax-jittable.

        For the `Fock` space this returns `x`, but for other hilbert spaces
        such as `Spin` this returns an array of indices.

        .. warning::
            This function is experimental. Use at your own risk.

        Args:
            x: a tensor containing samples from this hilbert space

        Returns:
            a tensor containing integer indices into the local hilbert
        """
        raise NotImplementedError(
            "states_to_local_indices(self, x) is not "
            f"implemented for Hilbert space {self} of type {type(self)}"
        )

    def local_indices_to_states(self, x: Array, dtype: DType = None):
        r"""
        Converts a tensor of integers to the corresponding local_values in
        this hilbert space.

        Equivalent to

        .. code::py

            hilbert.local_states[x]

        The input last dimension must match the size of this Hilbert space.
        This function can be jax-jitted.

        Args:
            x: a tensor with integer dtype and whose last dimension matches
                the size of this Hilbert space.

        Returns:
            a tensor with the same shape as the input, and values corresponding
            to the local_state indexed by the input tensor `x`.
        """
        raise NotImplementedError()

    @property
    def is_indexable(self) -> bool:
        """Whether the space can be indexed with an integer"""
        if not self.is_finite:
            return False
        return is_indexable(self.shape)

    def __mul__(self, other: "DiscreteHilbert"):
        if type(self) == type(other):
            res = self._mul_sametype_(other)
            if res is not NotImplemented:
                return res

        if isinstance(other, DiscreteHilbert):
            from .tensor_hilbert_discrete import TensorDiscreteHilbert

            return TensorDiscreteHilbert(self, other)
        elif isinstance(other, AbstractHilbert):
            from .tensor_hilbert import TensorGenericHilbert

            return TensorGenericHilbert(self, other)

        return NotImplemented

    def __pow__(self, n):
        return reduce(lambda x, y: x * y, [self for _ in range(n)])
