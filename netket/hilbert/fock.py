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

from typing import List, Tuple, Optional, Union, Iterable

import jax
from jax import numpy as jnp
import numpy as np
from numba import jit

from netket.graph import AbstractGraph

from .custom_hilbert import CustomHilbert
from ._deprecations import graph_to_N_depwarn


@jit(nopython=True)
def _sum_constraint(x, n_particles):
    return np.sum(x, axis=1) == n_particles


class Fock(CustomHilbert):
    r"""Hilbert space obtained as tensor product of local fock basis."""

    def __init__(
        self,
        n_max: Optional[int] = None,
        N: int = 1,
        n_particles: Optional[int] = None,
        graph: Optional[AbstractGraph] = None,
    ):
        r"""
        Constructs a new ``Boson`` given a maximum occupation number, number of sites
        and total number of bosons.

        Args:
          n_max: Maximum occupation for a site (inclusive). If None, the local occupation
            number is unbounded.
          N: number of bosonic modes (default = 1)
          n_particles: Constraint for the number of particles. If None, no constraint
            is imposed.
          graph: (Deprecated, pleaese use `N`) A graph, from which the number of nodes is extracted.

        Examples:
           Simple boson hilbert space.

           >>> from netket.hilbert import Boson
           >>> hi = Boson(n_max=5, n_particles=11, N=3)
           >>> print(hi.size)
           3
        """
        N = graph_to_N_depwarn(N=N, graph=graph)

        self._n_max = n_max

        if n_particles is not None:
            n_particles = int(n_particles)
            self._n_particles = n_particles
            assert n_particles > 0

            if self._n_max is None:
                self._n_max = n_particles
            else:
                if self._n_max * N < n_particles:
                    raise Exception(
                        """The required total number of bosons is not compatible
                        with the given n_max."""
                    )

            def constraints(x):
                return _sum_constraint(x, n_particles)

        else:
            constraints = None
            self._n_particles = None

        if self._n_max is not None:
            # assert self._n_max > 0
            local_states = np.arange(self._n_max + 1)
        else:
            max_ind = np.iinfo(np.intp).max
            self._n_max = max_ind
            local_states = None

        self._hilbert_index = None

        super().__init__(local_states, N, constraints)

    @property
    def n_max(self) -> Optional[int]:
        r"""The maximum number of bosons per site, or None
        if the number is unconstrained."""
        return self._n_max

    @property
    def n_particles(self) -> Optional[int]:
        r"""The total number of particles, or None
        if the number is unconstrained."""
        return self._n_particles

    def __pow__(self, n) -> "Fock":
        if self.n_particles is None:
            n_particles = None
        else:
            n_particles = n_particles * n

        return Fock(self.n_max, self.size * n, n_particles=n_particles)

    def _mul_sametype_(self, other: "Fock") -> "Fock":
        assert type(self) == type(other)
        if self.n_max == other.n_max:
            if self._n_particles is None and other._n_particles is None:
                return Fock(self.n_max, N=self.size + other.size)

        return NotImplemented

    def ptrace(self, sites: Union[int, List]) -> Optional["Fock"]:
        if isinstance(sites, int):
            sites = [sites]

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        if self.n_particles is not None:
            raise TypeError(
                "Cannot take the partial trace with a total particles constraint."
            )

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            return Fock(self.n_max, N=self.size - Nsites)

    def __repr__(self):
        n_particles = (
            ", n_particles={}".format(self._n_particles)
            if self._n_particles is not None
            else ""
        )
        nmax = self._n_max if self._n_max < np.iinfo(np.intp).max else "INT_MAX"
        return "Fock(n_max={}{}, N={})".format(nmax, n_particles, self._size)

    @property
    def _attrs(self):
        return (self.size, self._n_max, self._constraint_fn)

    def _random_state_with_constraint_legacy(self, out, rgen, n_max):
        sites = list(range(self.size))

        out.fill(0.0)
        ss = self.size

        for i in range(self.n_particles):
            s = rgen.integers(0, ss, size=())

            out[sites[s]] += 1

            if out[sites[s]] == n_max:
                sites.pop(s)
                ss -= 1

    def _random_state_legacy(self, size=None, *, out=None, rgen=None):
        if isinstance(size, int):
            size = (size,)
        shape = (*size, self._size) if size is not None else (self._size,)

        if out is None:
            out = np.empty(shape=shape)

        if rgen is None:
            rgen = np.random.default_rng()

        if self.n_particles is None:
            out[:] = rgen.integers(0, self.n_max, size=shape)
        else:
            if size is not None:
                out_r = out.reshape(-1, self._size)
                for b in range(out_r.shape[0]):
                    self._random_state_with_constraint_legacy(
                        out_r[b], rgen, self.n_max
                    )
            else:
                self._random_state_with_constraint_legacy(out, rgen, self.n_max)

        return out


from netket.utils import deprecated


@deprecated(
    """
Boson has been replaced by Fock. Please use fock, which has
the same semantics except for n_bosons which was replaced by
n_particles.

You should update your code because it will break in a future
version.
"""
)
def Boson(
    n_max: Optional[int] = None,
    N: int = 1,
    n_bosons: Optional[int] = None,
    graph: Optional[AbstractGraph] = None,
) -> Fock:
    return Fock(n_max, N, n_bosons, graph)
