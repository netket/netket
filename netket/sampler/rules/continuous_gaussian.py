import jax
import jax.numpy as jnp

from flax import struct


from ..metropolis import MetropolisRule


@struct.dataclass
class GaussianRule(MetropolisRule):
    r"""
    A transition rule acting on all particle positions at once.

    New proposals of particle positions are generated according to a Gaussian distribution of width sigma.
    """
    sigma: float = 1.0

    def transition(rule, sampler, machine, parameters, state, key, r):

        n_chains = r.shape[0]
        hilb = sampler.hilbert

        boundary = jnp.tile(hilb.pbc_to_array(), (n_chains, 1))
        modulus = hilb.L_to_array()

        prop = jax.random.normal(key, shape=(n_chains, hilb.size)) * rule.sigma
        rp = jnp.where(jnp.equal(boundary, False), r + prop, (r + prop) % modulus)

        return rp, None

    def __repr__(self):
        return "GaussianRule(sigma={})".format(self.sigma)
