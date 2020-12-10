from .custom_hilbert import CustomHilbert

from fractions import Fraction

import numpy as _np
from netket import random as _random
from numba import jit

from typing import Optional, List

import jax


class Spin(CustomHilbert):
    r"""Hilbert space obtained as tensor product of local spin states."""

    def __init__(self, s: float, N: int = 1, total_sz: Optional[float] = None):
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
        local_size = round(2 * s + 1)
        local_states = _np.empty(local_size)

        assert int(2 * s + 1) == local_size

        for i in range(local_size):
            local_states[i] = -round(2 * s) + 2 * i
        local_states = local_states.tolist()

        self._check_total_sz(total_sz, N)
        if total_sz is not None:

            def constraints(x):
                return self._sum_constraint(x, total_sz)

        else:
            constraints = None

        self._total_sz = total_sz if total_sz is None else int(total_sz)
        self._s = s
        self._local_size = local_size

        super().__init__(local_states, N, constraints)

    def random_state(self, out=None, rgen=None):
        r"""Member function generating uniformely distributed local random states.

        Args:
            out: If provided, the random quantum numbers will be inserted into this array.
                 It should be of the appropriate shape and dtype.
            rgen: The random number generator. If None, the global
                  NetKet random number generator is used.

        Examples:
           Test that a new random state is a possible state for the hilbert
           space.

           >>> import netket as nk
           >>> import numpy as np
           >>> hi = nk.hilbert.Spin(N=4)
           >>> rstate = np.zeros(hi.size)
           >>> hi.random_state(rstate)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True
        """

        if out is None:
            out = _np.empty(self._size)

        if rgen is None:
            rgen = _random

        if self._total_sz is None:
            for i in range(self._size):
                rs = rgen.randint(0, self._local_size)
                out[i] = self.local_states[rs]
        else:
            sites = list(range(self.size))

            out.fill(-round(2 * self._s))
            ss = self.size

            for i in range(round(self._s * self.size) + self._total_sz):
                s = rgen.randint(0, ss)

                out[sites[s]] += 2

                if out[sites[s]] > round(2 * self._s - 1):
                    sites.pop(s)
                    ss -= 1

        return out

    @staticmethod
    @jit(nopython=True)
    def _sum_constraint(x, total_sz):
        return _np.sum(x, axis=1) == round(2 * total_sz)

    def _check_total_sz(self, total_sz, size):
        if total_sz is None:
            return

        m = round(2 * total_sz)
        if _np.abs(m) > size:
            raise Exception(
                "Cannot fix the total magnetization: 2|M| cannot " "exceed Nspins."
            )

        if (size + m) % 2 != 0:
            raise Exception(
                "Cannot fix the total magnetization: Nspins + " "totalSz must be even."
            )

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

    def __jax_random_state__(self, key, dtype):
        return self.__jax_random_state_batch__(key, dtype)[0, :]

    def __jax_random_state_batch__(self, key, size, dtype):
        if self._total_sz is None:
            return _jax_randstate_fast_(key, (size, self.size), dtype, self._s)
        else:
            return _jax_randstate_fast_constrained_(
                key, (size, self.size), dtype, self._s, self._total_sz
            )

    def __jax_flip_state__(self, key, state, index):
        if self._s == 0.5:
            return _jax_flipat_N2_(key, state, index)
        else:
            return _jax_flipat_generic_(key, state, index, self._s)


import jax
from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3))
def _jax_randstate_fast_(key, shape, dtype, s):

    n_states = int(2 * s + 1)

    rs = jax.random.randint(key, shape=shape, minval=0, maxval=n_states)

    return jax.numpy.array(rs * 2.0 - (n_states - 1), dtype=dtype)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _jax_randstate_fast_constrained_(key, shape, dtype, s, total_sz):
    batches = shape[0]
    N = shape[1]

    n_states = int(2 * s + 1)
    assert N == 2, "Fast contrained sapling is only supported for spin 1/2"
    m = total_sz * 2
    nup = (N + m) // 2
    ndown = (N - m) // 2

    x = jax.numpy.array(
        np.concatenate((np.ones((batches, nup)), -np.ones((batches, ndown)))),
        dtype=dtype,
    )

    return jax.random.shuffle(key, x, axis=1)


def _jax_flipat_generic_(key, x, i, s):
    dtype = x.dtype
    n_states = int(2 * s + 1)

    xi_old = x[i]
    r = jax.random.uniform(key)
    xi_new = jax.numpy.floor(r * (n_states - 1)) * 2 - (n_states - 1)
    xi_new = xi_new + 2 * (xi_new >= xi_old)

    return jax.ops.index_update(x, i, xi_new), xi_old


def _jax_flipat_N2_(key, x, i):
    return jax.ops.index_update(x, i, -x[i]), x[i]
