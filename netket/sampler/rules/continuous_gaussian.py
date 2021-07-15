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

        pbc = jnp.array(hilb.n_particles * hilb.pbc)
        boundary = jnp.tile(pbc, (n_chains, 1))

        Ls = jnp.array(hilb.n_particles * hilb.extend)
        modulus = jnp.where(jnp.equal(pbc, False), jnp.inf, Ls)

        prop = jax.random.normal(key, shape=(n_chains, hilb.size)) * rule.sigma
        rp = jnp.where(jnp.equal(boundary, False), r + prop, (r + prop) % modulus)

        return rp, None

    def __repr__(self):
        return "GaussianRule(sigma={})".format(self.sigma)
