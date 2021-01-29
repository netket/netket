from fractions import Fraction
from typing import Optional, List

import jax
from jax import numpy as jnp
import numpy as np
from netket.graph import AbstractGraph
from numba import jit


from .custom_hilbert import CustomHilbert
from ._deprecations import graph_to_N_depwarn


def _check_total_sz(total_sz, size):
    if total_sz is None:
        return

    m = round(2 * total_sz)
    if np.abs(m) > size:
        raise Exception(
            "Cannot fix the total magnetization: 2|M| cannot " "exceed Nspins."
        )

    if (size + m) % 2 != 0:
        raise Exception(
            "Cannot fix the total magnetization: Nspins + " "totalSz must be even."
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

        _check_total_sz(total_sz, N)
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

    def __repr__(self):
        total_sz = (
            ", total_sz={}".format(self._total_sz) if self._total_sz is not None else ""
        )
        return "Spin(s={}{}, N={})".format(Fraction(self._s), total_sz, self._size)
