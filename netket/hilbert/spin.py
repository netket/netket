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

from fractions import Fraction
from typing import Optional, List, Union, Iterable

import jax
from jax import numpy as jnp
import numpy as np
from netket.graph import AbstractGraph
from numba import jit


from .custom_hilbert import CustomHilbert
from ._deprecations import graph_to_N_depwarn


def _check_total_sz(total_sz, S, size):
    if total_sz is None:
        return

    local_size = 2 * S + 1

    m = round(2 * total_sz)
    if np.abs(m) > size * (2 * S):
        raise ValueError(
            "Cannot fix the total magnetization: 2|M| cannot " "exceed Nspins."
        )

    # If half-integer spins (1/2, 3/2)
    if local_size % 2 == 0:
        # Check that the total magnetization is odd if odd spins or even if even # of spins
        if (size + m) % 2 != 0:
            raise ValueError(
                "Cannot fix the total magnetization: Nspins + 2*totalSz must be even."
            )
    # else if full-integer (S=1,2)
    else:
        if m % 2 != 0:
            raise ValueError(
                "Cannot fix the total magnetization to a half-integer number"
            )


@jit(nopython=True)
def _sum_constraint(x, total_sz):
    return np.sum(x, axis=1) == round(2 * total_sz)


class Spin(CustomHilbert):
    r"""Hilbert space obtained as tensor product of local spin states."""

    def __init__(
        self,
        s: float,
        N: int = 1,
        total_sz: Optional[float] = None,
        graph: Optional[AbstractGraph] = None,
    ):
        r"""Hilbert space obtained as tensor product of local spin states.

        Args:
           s: Spin at each site. Must be integer or half-integer.
           N: Number of sites (default=1)
           total_sz: If given, constrains the total spin of system to a particular value.
           graph: (deprecated) a graph from which to extract the number of sites.

        Examples:
           Simple spin hilbert space.

           >>> from netket.hilbert import Spin
           >>> g = Hypercube(length=10,n_dim=2,pbc=True)
           >>> hi = Spin(s=0.5, N=4)
           >>> print(hi.size)
           4
        """
        N = graph_to_N_depwarn(N=N, graph=graph)

        local_size = round(2 * s + 1)
        local_states = np.empty(local_size)

        assert int(2 * s + 1) == local_size

        for i in range(local_size):
            local_states[i] = -round(2 * s) + 2 * i
        local_states = local_states.tolist()

        _check_total_sz(total_sz, s, N)
        if total_sz is not None:

            def constraints(x):
                return _sum_constraint(x, total_sz)

        else:
            constraints = None

        self._total_sz = total_sz if total_sz is None else int(total_sz)
        self._s = s
        self._local_size = local_size

        super().__init__(local_states, N, constraints)

    def __pow__(self, n):
        if self._total_sz is None:
            total_sz = None
        else:
            total_sz = total_sz * n

        return Spin(self._s, self.size * n, total_sz=total_sz)

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        if self._s == other._s:
            if self._total_sz is None and other._total_sz is None:
                return Spin(s=self._s, N=self.size + other.size)

        return NotImplemented

    def ptrace(self, sites: Union[int, List]) -> Optional["Spin"]:
        if isinstance(sites, int):
            sites = [sites]

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        if self._total_sz is not None:
            raise TypeError(
                "Cannot take the partial trace with a total magnetization constraint."
            )

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            return Spin(s=self._s, N=self.size - Nsites)

    def __repr__(self):
        total_sz = (
            ", total_sz={}".format(self._total_sz) if self._total_sz is not None else ""
        )
        return "Spin(s={}{}, N={})".format(Fraction(self._s), total_sz, self._size)

    @property
    def _attrs(self):
        return (self.size, self._s, self._constraint_fn)

    def _random_state_legacy(self, size=None, *, out=None, rgen=None):
        if isinstance(size, int):
            size = (size,)
        shape = (*size, self._size) if size is not None else (self._size,)

        if out is None:
            out = np.empty(shape=shape)

        if rgen is None:
            rgen = np.random.default_rng()

        if self._total_sz is None:
            out[:] = rgen.choice(self.local_states, size=shape)
        else:
            # TODO: this can most likely be done more efficiently
            if size is not None:
                out_r = out.reshape(-1, self._size)
                for b in range(out_r.shape[0]):
                    self._random_state_with_constraint_legacy(out_r[b], rgen)
            else:
                self._random_state_with_constraint_legacy(out, rgen)

        return out

    def _random_state_with_constraint_legacy(self, out, rgen):
        sites = list(range(self.size))
        out.fill(-round(2 * self._s))
        ss = self.size

        for i in range(round(self._s * self.size) + self._total_sz):
            s = rgen.integers(0, ss, size=())

            out[sites[s]] += 2

            if out[sites[s]] > round(2 * self._s - 1):
                sites.pop(s)
                ss -= 1
