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

from typing import Optional, List, Callable

from numbers import Real

from netket.graph import AbstractGraph

from .homogeneous import HomogeneousHilbert
from ._deprecations import graph_to_N_depwarn


class CustomHilbert(HomogeneousHilbert):
    r"""A custom hilbert space with discrete local quantum numbers."""

    def __init__(
        self,
        local_states: Optional[List[Real]],
        N: int = 1,
        constraint_fn: Optional[Callable] = None,
        graph: Optional[AbstractGraph] = None,
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
        N = graph_to_N_depwarn(N=N, graph=graph)

        super().__init__(local_states, N, constraint_fn)

    def __pow__(self, n):
        if self._has_constraint:
            raise NotImplementedError(
                """Cannot exponentiate a CustomHilbert with constraints.
                Construct it from scratch instead."""
            )

        return CustomHilbert(self._local_states, self.size * n)
