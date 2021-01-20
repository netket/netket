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

    def _random_state_batch_impl(hilb, key, batches, dtype):
        S = hilb._s
        shape = (batches, hilb.size)

        # If unconstrained space, use fast sampling
        if hilb._total_sz is None:
            n_states = int(2 * S + 1)
            rs = jax.random.randint(key, shape=shape, minval=0, maxval=n_states)

            two = jnp.asarray(2, dtype=dtype)
            return jnp.asarray(rs * two - (n_states - 1), dtype=dtype)
        else:
            N = hilb.size
            n_states = int(2 * S) + 1
            # if constrained and S == 1/2, use a trick to sample quickly
            if n_states == 2:
                m = hilb._total_sz * 2
                nup = (N + m) // 2
                ndown = (N - m) // 2

                x = jnp.concatenate(
                    (
                        jnp.ones((batches, nup), dtype=dtype),
                        -jnp.ones(
                            (
                                batches,
                                ndown,
                            ),
                            dtype=dtype,
                        ),
                    ),
                    axis=1,
                )

                # deprecated: return jax.random.shuffle(key, x, axis=1)
                return jax.vmap(jax.random.permutation)(
                    jax.random.split(key, x.shape[0]), x
                )

            # if constrained and S != 1/2, then use a slow fallback algorithm
            # TODO: find better, faster way to smaple constrained arbitrary spaces.
            else:
                from jax.experimental import host_callback as hcb

                cb = lambda rng: _random_states_with_constraint(
                    hilb, rng, batches, dtype
                )

                state = hcb.call(
                    cb,
                    key,
                    result_shape=jax.ShapeDtypeStruct(shape, dtype),
                )

                return state

        return out

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


# TODO: could numba-jit this
def _random_states_with_constraint(hilb, rngkey, n_batches, dtype):
    out = np.full((n_batches, hilb.size), -round(2 * hilb._s), dtype=dtype)
    rgen = np.random.default_rng(rngkey)

    for b in range(n_batches):
        sites = list(range(hilb.size))
        ss = hilb.size

        for i in range(round(hilb._s * hilb.size) + hilb._total_sz):
            s = rgen.integers(0, ss, size=())

            out[b, sites[s]] += 2

            if out[b, sites[s]] > round(2 * hilb._s - 1):
                sites.pop(s)
                ss -= 1

    return out
