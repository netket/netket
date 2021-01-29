import jax
import flax
import numpy as np

from jax import numpy as jnp
from flax import struct

from netket.hilbert.random import flip_state

from ..metropolis import MetropolisRule


@struct.dataclass
class LocalRule(MetropolisRule):
    def transition(rule, sampler, machine, parameters, state, key, σ):
        key1, key2 = jax.random.split(key, 2)

        n_chains = σ.shape[0]
        hilb = sampler.hilbert
        local_states = jnp.array(np.sort(np.array(hilb.local_states)))

        indxs = jax.random.randint(key1, shape=(n_chains,), minval=0, maxval=hilb.size)
        σp, _ = flip_state(hilb, key2, σ, indxs)

        return σp, None
