import jax
import jax.numpy as jnp

from flax import struct


from ..metropolis import MetropolisRule

@struct.dataclass
class GaussianRule(MetropolisRule):
    r"""
    A transition rule acting on all particle positions at once.

    New proposals of particle positions are generated according to a Gaussian distribution of width s.
    """
    s: float = 1.

    def transition(rule, sampler, machine, parameters, state, key, σ, sigma):

        n_chains = σ.shape[0]
        hilb = sampler.hilbert
        boundary = jnp.tile(jnp.array(hilb.N * hilb.L))
        modulus = jnp.where(boundary == None, jnp.inf, boundary)

        prop = jax.random.normal(key, shape=(n_chains, hilb.size)) * rule.sigma
        σp = jnp.where(boundary == None, σ + prop, (σ + prop) % modulus)

        return σp, None

    def __repr__(self):
        return "GaussianRule()"
