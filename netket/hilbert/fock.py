from typing import List, Tuple, Optional

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
            assert self._n_max > 0
            local_states = np.arange(self._n_max + 1)
        else:
            max_ind = np.iinfo(np.intp).max
            self._n_max = max_ind
            local_states = None

        self._hilbert_index = None

        super().__init__(local_states, N, constraints)

    @property
    def n_max(self):
        r"""int or None: The maximum number of bosons per site, or None
        if the number is unconstrained."""
        return self._n_max

    @property
    def n_particles(self):
        r"""int or None: The total number of particles, or None
        if the number is unconstrained."""
        return self._n_particles

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
):
    return Fock(n_max, N, n_bosons, graph)
