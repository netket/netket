import jax
import numpy as np

from flax import struct
from numba4jax import njit4jax

from ..metropolis import MetropolisRule


@struct.dataclass
class BetterRule(MetropolisRule):
    r"""
    A transition rule acting on the local degree of freedom.

    This transition acts locally only on one local degree of freedom :math:`s_i`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s_N`,
    where :math:`s^\prime_i \neq s_i`.

    The transition probability associated to this
    sampler can be decomposed into two steps:

    1. One of the site indices :math:`i = 1\dots N` is chosen
    with uniform probability.
    2. Among all the possible (:math:`m`) values that :math:`s_i` can take,
    one of them is chosen with uniform probability.
    """

    def transition(rule, sampler, machine, parameters, state, key, σ):
        key1, key2 = jax.random.split(key, 2)

        n_chains = σ.shape[0]
        N = sampler.hilbert.physical.size

        # cases
        # 1-Only left
        # 2-Only right
        # 3-both
        # 4-
        move_probs = np.cumsum(np.array([0.225, 0.225, 0.5, 0.05]))

        # r = jax.random.uniform(key1, shape=(n_chains, 1))
        # move_id = jnp.sum(r>move_probs, axis=1)

        # def move_left(args):
        #    σ, = args
        #    return flip_state(hilb, key2, σ, indxs)

        @njit4jax((jax.abstract_arrays.ShapedArray(σ.shape, σ.dtype),))
        def _transition(args):
            # unpack arguments
            v_proposed, v, rand_vec = args

            v_proposed[:, :] = v[:, :]
            rand_i = 0

            for i in range(n_chains):
                r = rand_vec[rand_i]
                rand_i = rand_i + 1
                if r < move_probs[0]:
                    # single left
                    site = int(np.floor(rand_vec[rand_i] * N))
                    rand_i = rand_i + 1
                    v_proposed[i, site] = -v_proposed[i, site]
                elif r < move_probs[1]:
                    # single right
                    site = int(np.floor(rand_vec[rand_i] * N)) + N
                    rand_i = rand_i + 1
                    v_proposed[i, site] = -v_proposed[i, site]
                elif r < move_probs[2]:
                    # both
                    site_l = int(np.floor(rand_vec[rand_i] * N))
                    rand_i = rand_i + 1
                    site_r = int(np.floor(rand_vec[rand_i] * N)) + N
                    rand_i = rand_i + 1
                    v_proposed[i, site_l] = -v_proposed[i, site_l]
                    v_proposed[i, site_r] = -v_proposed[i, site_r]
                else:
                    rr = rand_vec[rand_i]
                    rand_i = rand_i + 1
                    if rr < 0.5:
                        v_proposed[i, :N] = v_proposed[i, N:]
                    else:
                        v_proposed[i, N:] = v_proposed[i, :N]

        rand_vec = jax.random.uniform(key, shape=(σ.shape[0] * 3,))

        (σp,) = _transition(σ, rand_vec)

        return σp, None

    def __repr__(self):
        return "BetterRule()"
