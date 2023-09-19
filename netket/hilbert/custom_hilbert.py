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

from numbers import Real

import jax.numpy as jnp

from .homogeneous import HomogeneousHilbert


class CustomHilbert(HomogeneousHilbert):
    r"""A custom hilbert space with discrete local quantum numbers."""

    def __init__(
        self,
        local_states: Optional[list[Real]],
        N: int = 1,
        constraint_fn: Optional[Callable] = None,
    ):
        r"""
        Constructs a new ``CustomHilbert`` given a list of eigenvalues of the states and
        a number of sites, or modes, within this hilbert space.

        Args:
            local_states (list or None): Eigenvalues of the states. If the allowed
                states are an infinite number, None should be passed as an argument.
            N: Number of modes in this hilbert space (default 1).
            constraint_fn: A function specifying constraints on the quantum numbers.
                Given a batch of quantum numbers it should return a vector
                of bools specifying whether those states are valid or not.

        Examples:
           Simple custom hilbert space.

           >>> import netket
           >>> g = netket.graph.Hypercube(length=10,n_dim=2,pbc=True)
           >>> hi = netket.hilbert.CustomHilbert(local_states=[-1232, 132, 0], N=100)
           >>> print(hi.size)
           100
        """
        super().__init__(local_states, N, constraint_fn)

    def states_to_local_indices(self, x):
        local_states = jnp.asarray(self.local_states)
        local_states = local_states.reshape(tuple(1 for _ in range(x.ndim)) + (-1,))
        x = x.reshape(x.shape + (1,))
        x_idmap = x == local_states
        idxs = jnp.arange(self.local_size).reshape(local_states.shape)
        return jnp.sum(x_idmap * idxs, axis=-1)

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        if not self.constrained:
            if self.local_states == other.local_states:
                return CustomHilbert(self._local_states, self.size + other.size)

        return NotImplemented
