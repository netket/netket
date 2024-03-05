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

from typing import Optional, Callable

import numpy as np

from netket.utils import StaticRange
from netket.utils.types import Array

from .discrete_hilbert import DiscreteHilbert
from .index import (
    HilbertIndex,
    UniformTensorProductHilbertIndex,
    optimalConstrainedHilbertindex,
)


class HomogeneousHilbert(DiscreteHilbert):
    r"""The Abstract base class for homogeneous hilbert spaces.

    This class should only be subclassed and should not be instantiated directly.

    .. note::

        To override the logic to index into constrained hilbert spaces, it is
        possible to use an informal interface built on top of non-public
        indexing classes.

        In particular, you can override the following properties and methods:

        - Do not specify the :code:`constraint_fn` keyword argument when
          calling the init method of this abstract class.
        - Override the property :py:attr:`~nk.hilbert.HomogeneousHilbert.constrained`,
          to return `True` or `False` depending on your own logic.
        - Override the property :py:attr:`~nk.hilbert.HomogeneousHilbert._hilbert_index`
          to return an hilbert index object (see the discussion in the source code of
          the folder :code:`netket/hilbert/index/__init__.py`).

    """

    def __init__(
        self,
        local_states: Optional[StaticRange],
        N: int = 1,
        constraint_fn: Optional[Callable] = None,
    ):
        r"""
        Constructs a new :class:`~netket.hilbert.HomogeneousHilbert` given a list of
        eigenvalues of the states and a number of sites, or modes, within this
        hilbert space.

        This method should only be called from the subclasses `__init__` method.

        Args:
            local_states: :class:`~netket.utils.StaticRange` object describing the
                numbers used to encode the local degree of freedom of this Hilbert
                Space.
            N: Number of modes in this hilbert space (default 1).
            constraint_fn: A function specifying constraints on the quantum numbers.
                Given a batch of quantum numbers it should return a vector of bools
                specifying whether those states are valid or not.
        """
        assert isinstance(N, int)

        if not (isinstance(local_states, StaticRange) or local_states is None):
            raise TypeError("local_states must be a StaticRange.")

        self._is_finite = local_states is not None

        if self._is_finite:
            self._local_states = local_states
            self._local_size = len(local_states)
        else:
            self._local_states = None
            self._local_size = np.iinfo(np.intp).max

        self._constraint_fn = constraint_fn

        self._hilbert_index_ = None

        shape = tuple(self._local_size for _ in range(N))
        super().__init__(shape=shape)

    @property
    def size(self) -> int:
        r"""The total number number of degrees of freedom."""
        return len(self.shape)

    @property
    def local_size(self) -> int:
        r"""Size of the local degrees of freedom that make the total hilbert space."""
        return self._local_size

    def size_at_index(self, i: int) -> int:
        return self.local_size

    @property
    def local_states(self) -> Optional[list[float]]:
        r"""A list of discrete local quantum numbers.
        If the local states are infinitely many, None is returned."""
        if self.is_finite:
            return list(self._local_states)
        return self._local_states

    def states_at_index(self, i: int):
        return self.local_states

    @property
    def n_states(self) -> int:
        r"""The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""
        if not self.is_indexable:
            raise RuntimeError("The hilbert space is too large to be indexed.")
        return self._hilbert_index.n_states

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
        return self._local_states.states_to_numbers(x, dtype=np.int32)

    @property
    def is_finite(self) -> bool:
        r"""Whether the local hilbert space is finite."""
        return self._is_finite

    @property
    def constrained(self) -> bool:
        r"""The hilbert space does not contain `prod(hilbert.shape)`
        basis states.

        Typical constraints are population constraints (such as fixed
        number of bosons, fixed magnetization...) which ensure that
        only a subset of the total unconstrained space is populated.

        Typically, objects defined in the constrained space cannot be
        converted to QuTiP or other formats.
        """
        return self._constraint_fn is not None

    def _numbers_to_states(self, numbers: np.ndarray) -> np.ndarray:
        return self._hilbert_index.numbers_to_states(numbers)

    def _states_to_numbers(self, states: np.ndarray):
        return self._hilbert_index.states_to_numbers(states)

    def all_states(self) -> np.ndarray:
        r"""Returns all valid states of the Hilbert space.

        Throws an exception if the space is not indexable.

        Returns:
            A (n_states x size) batch of states. this corresponds
            to the pre-allocated array if it was passed.
        """
        if not self.is_indexable:  # includes call to _setup
            raise RuntimeError("The hilbert space is too large to be indexed.")

        return self._hilbert_index.all_states()

    @property
    def _hilbert_index(self) -> HilbertIndex:
        """
        The `self._hilbert_index` is a lazily constructed object used to index into homogeneous Hilbert spaces.

        This indexing object implements the logic for `number_to_states`, `states_to_numbers` and `n_states`,
        as well as the handling of constraints if necessary.
        """
        if self._hilbert_index_ is None:
            if not self.constrained:
                index = UniformTensorProductHilbertIndex(self._local_states, self.size)
            else:
                index = optimalConstrainedHilbertindex(
                    self._local_states, self.size, self._constraint_fn
                )
            self._hilbert_index_ = index
        return self._hilbert_index_

    @property
    def is_indexable(self) -> bool:
        """Whether the space can be indexed with an integer"""
        if not self.is_finite:
            return False
        return self._hilbert_index.is_indexable

    def __repr__(self):
        constr = f", constrained={self.constrained}" if self.constrained else ""

        clsname = type(self).__name__
        return f"{clsname}(local_size={self._local_size}, N={self.size}{constr})"

    @property
    def _attrs(self):
        return (
            self.size,
            self.local_size,
            self._local_states,
            self.constrained,
            self._constraint_fn,
        )
