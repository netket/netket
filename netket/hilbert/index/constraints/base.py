# Copyright 2023-2024 The NetKet Authors - All rights reserved.
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

from functools import partial
from typing import Callable
import abc

import numpy as np

import jax
import jax.numpy as jnp

from typing import Union
from netket.utils.types import Array
from netket.utils import struct, StaticRange
from netket.utils.dispatch import dispatch

from ..base import HilbertIndex
from ..uniform_tensor import UniformTensorProductHilbertIndex


class DiscreteHilbertConstraint(struct.Pytree):
    r"""
    Protocol to define an Abstract Constraint for a discete Hilbert space.

    To define a customized constraint, you must subclass this class and at least implement the
    :code:`__call__` method. The :code:`__call__` method should take as input a matrix encoding a batch of
    configurations, and return a vector of booleans specifying whether they are valid configurations
    or not.

    The :code:`__call__` method must be :code:`jax.jit`-able. If you cannot make it jax-jittable, you can implement
    it in numba/python and wrap it into a :func:`jax.pure_callback` to make it compatible with jax.

    The callback should be hashable and comparable with itself, which means it must implement :code:`__hash__` and :code:`__eq__`.
    By default, the :code:`__hash__` method is implemented by the `id` of the object, which is unique for each object,
    which will work but might lead to more recompilations in jax. If you can, you should implement a custom :code:`__hash__`

    Example:

        The following example shows a class that implements a simple constraint checking that the total sum of the
        elements in the configuration is equal to a given value. The example shows how to implement the :code:`__call__` method
        and the :code:`__hash__` and :code:`__eq__` methods.

        .. code-block:: python

            import netket as nk
            import jax; import jax.numpy as jnp

            class SumConstraint(nk.hilbert.DiscreteHilbertConstraint):
                # A simple constraint checking that the total sum of the elements
                # in the configuration is equal to a given value.

                def __init__(self, total_sum):
                    self.total_sum = total_sum

                def __call__(self, x):
                    # Makes it jax-compatible
                   return jnp.sum(x, axis=-1) == self.total_sum

                def __hash__(self):
                    return hash(("SumConstraint", self.total_sum))

                def __eq__(self, other):
                    if isinstance(other, SumConstraint):
                        return self.total_sum == other.total_sum
                    return False

    Example:

        The following example shows how to implement the same function as above, but using a pure python function and
        a :func:`jax.pure_callback` to make it compatible with jax.

        .. code-block:: python

            import netket as nk
            import jax; import jax.numpy as jnp

            class SumConstraintPy(nk.hilbert.DiscreteHilbertConstraint):
                # A simple constraint checking that the total sum of the elements
                # in the configuration is equal to a given value.

                def __init__(self, total_sum):
                    self.total_sum = total_sum

                def __call__(self, x):
                    return jax.pure_callback(self._call_py, result_shape_dtypes=(jax.ShapeDtypeStruct(x.shape[:-1], bool), x, vectorized=True)

                def _call_py(self, x):
                    # Not Jax compatible
                    return np.sum(x, axis=-1) == self.total_sum

                def __hash__(self):
                    return hash(("SumConstraint", self.total_sum))

                def __eq__(self, other):
                    if isinstance(other, SumConstraint):
                        return self.total_sum == other.total_sum
                    return False

    """

    @abc.abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        This function should take as input a matrix encoding a batch of configurations,
        and return a vector of booleans specifying whether they are valid configurations of
        the Hilbert space or not.

        Args:
            x: 2D matrix.
        """


@dispatch.abstract
def optimalConstrainedHilbertindex(local_states, size, constraint):
    """
    Returns the optimal Hilbert Index to index into the uniform
    state with local degrees of freedom `local_states`, size sites
    and given constraint.

    This function uses dispatch to select a potential optimal implementation,
    and generally returns a default implementation if None better is available.

    Args:
        local_states: The StaticRange Local states.
        size: integer of the number of degrees of freedom.
        constraint: callable class implementing the constraint.
    """


@struct.dataclass
class ConstrainedHilbertIndex(HilbertIndex):
    """
    Indexes a constrained hilbert space with a generic constraint function,
    by building an internal lookup table of all states in the constrained space.

    Requires that the unconstrained index is indexable.
    """

    unconstrained_index: HilbertIndex
    constraint_fun: DiscreteHilbertConstraint = struct.field(pytree_node=False)

    @property
    def _bare_numbers(self) -> Array:
        return compute_constrained_to_bare_conversion_table(
            self.unconstrained_index, self.constraint_fun
        )

    @property
    def n_states(self) -> int:
        return self._bare_numbers.shape[0]

    @property
    def size(self) -> int:
        return self.unconstrained_index.size

    @property
    def local_states(self) -> Union[Array, StaticRange]:
        return self.unconstrained_index.local_states

    @property
    def local_size(self) -> int:
        return self.unconstrained_index.local_size

    @jax.jit
    def states_to_numbers(self, states: Array) -> Array:
        out = self.unconstrained_index.states_to_numbers(states)
        return jnp.searchsorted(self._bare_numbers, out)

    @jax.jit
    def numbers_to_states(self, numbers: Array) -> Array:
        # convert to original space
        numbers = self._bare_numbers[numbers]
        return self.unconstrained_index.numbers_to_states(numbers)

    def all_states(self) -> Array:
        return self.numbers_to_states(jnp.arange(self.n_states))

    @property
    def is_indexable(self) -> bool:
        return self.unconstrained_index.is_indexable

    @property
    def n_states_bound(self) -> int:
        """
        Returns an integer upper bound on the total number of states.
        Used as a proxy of the computational cost of using this indexer object,
        and by Hilbert spaces to decide which implementation to pick.
        """
        return self.unconstrained_index.n_states


@optimalConstrainedHilbertindex.dispatch
def optimalConstrainedHilbertindex_generic(local_states, size, constraint):
    # Generic dispatch rule based on a lookup table.
    bare_index = UniformTensorProductHilbertIndex(local_states, size)
    return ConstrainedHilbertIndex(bare_index, constraint)


# This function has exponential runtime in self.size, so we cache it in order to
# only compute it once.
# TODO: distribute over devices/MPI (expensive constraint_fun),  choose better chunk size
@partial(jax.jit, static_argnames=("chunk_size", "constraint_fun"))
def compute_constrained_to_bare_conversion_table(
    hilbert_index: HilbertIndex,
    constraint_fun: Callable[[Array], Array],
    *,
    chunk_size: int = 65536,
):
    """
    Computes the conversion table that converts the 'constrained' indices
    of an hilbert space to bare indices, so that routines generating
    only values in an unconstrained space can be used.

    Args:
        hilbert_index:
            A dataclass with only metadata (only pytree_node=False)
        constraint_fun:
            A dataclass with only metadata (only pytree_node=False) and __call__ attribute
            Python functions can be used by wrapping them in a jax.tree_util.Partial
            with no args and keywords.
        chunk_size: (optional, default=65536)
            This function operates on blocks of `chunk_size` states at a time in order
            to lower the memory cost. The default chunk size has been chosen arbitrarily
            and might need tweaking depending on the particular constraint_fun.
    """

    with jax.ensure_compile_time_eval():
        n_chunks = int(np.ceil(hilbert_index.n_states / chunk_size))
        bare_number_chunks = []
        for i in range(n_chunks):
            id_start = chunk_size * i
            id_end = np.minimum(chunk_size * (i + 1), hilbert_index.n_states)
            ids = jnp.arange(id_start, id_end, dtype=jnp.int32)
            states = hilbert_index.numbers_to_states(ids)
            is_constrained = constraint_fun(states)
            (chunk_bare_number,) = jnp.nonzero(is_constrained)
            bare_number_chunks.append(chunk_bare_number + id_start)
        bare_numbers = jnp.concatenate(bare_number_chunks)
    return bare_numbers
