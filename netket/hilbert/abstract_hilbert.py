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

from typing import Tuple, Optional, Union, Iterable

import jax.numpy as jnp
import numpy as np

max_states = np.iinfo(np.int32).max
"""int: Maximum number of states that can be indexed"""


class AbstractHilbert(abc.ABC):
    """Abstract class for NetKet hilbert objects.

    This class definese the common interface used to interact with Hilbert spaces.

    An AbstractHilbert object identifies an Hilbert space and a computational basis on
    such hilbert space, such as the z-basis for spins on a lattice, or the
    position-basis for particles in a box.

    Hilbert Spaces are generally immutable python objects that must be hashable in order
    to be used as static arguments to `jax.jit` functions.
    """

    def __init__(self):
        self._hash = None

    @property
    @abc.abstractmethod
    def size(self) -> int:
        r"""The number number of degrees of freedom in the basis of this
        Hilbert space."""
        raise NotImplementedError()  # pragma: no cover

    def random_state(
        self,
        key=None,
        size: Optional[int] = None,
        dtype=np.float32,
    ) -> jnp.ndarray:
        r"""Generates either a single or a batch of uniformly distributed random states.
        Runs as :code:`random_state(self, key, size=None, dtype=np.float32)` by default.

        Args:
            key: rng state from a jax-style functional generator.
            size: If provided, returns a batch of configurations of the form
                  :code:`(size, N)` if size is an integer or :code:`(*size, N)` if it is
                  a tuple and where :math:`N` is the Hilbert space size.
                  By default, a single random configuration with shape
                  :code:`(#,)` is returned.
            dtype: DType of the resulting vector.

        Returns:
            A state or batch of states sampled from the uniform distribution on the
            hilbert space.

        Example:

            >>> import netket, jax
            >>> hi = netket.hilbert.Qubit(N=2)
            >>> k1, k2 = jax.random.split(jax.random.PRNGKey(1))
            >>> print(hi.random_state(key=k1))
            [1. 0.]
            >>> print(hi.random_state(key=k2, size=2))
            [[0. 0.]
             [0. 1.]]
        """
        from netket.hilbert import random

        return random.random_state(self, key, size, dtype=dtype)

    def ptrace(self, sites: Union[int, Iterable]) -> "AbstractHilbert":
        """Returns the hilbert space without the selected sites.

        Not all hilbert spaces support this operation.

        Args:
            sites: a site or list of sites to trace away

        Returns:
            The partially-traced hilbert space. The type of the resulting hilbert space
            might be different from the starting one.
        """
        raise NotImplementedError("Ptrace not implemented for this hilbert space type.")

    @property
    def is_indexable(self) -> bool:
        """Whever the space can be indexed with an integer"""
        return False

    @property
    @abc.abstractmethod
    def _attrs(self) -> Tuple:
        """
        Tuple of hashable attributs, used to compute the immutable
        hash of this Hilbert space
        """

    def __eq__(self, other) -> bool:
        if isinstance(other, type(self)):
            return self._attrs == other._attrs

        return False

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._attrs)

        return self._hash
