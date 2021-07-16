import jax
from jax import numpy as jnp

from netket.hilbert import AbstractParticle
from netket.utils.dispatch import dispatch


@dispatch
def random_state(hilb: AbstractParticle, key, batches: int, *, dtype):
    """Positions particles w.r.t. normal distribution,
    if no periodic boundary conditions are applied
    in a spatial dimension. Otherwise the particles are
    positioned evenly along the box from 0 to L, with Gaussian noise
    of certain width."""
    pbc = jnp.array(hilb.n_particles * hilb.pbc)
    boundary = jnp.tile(pbc, (batches, 1))

    Ls = jnp.array(hilb.n_particles * hilb.extend)
    modulus = jnp.where(jnp.equal(pbc, False), jnp.inf, Ls)

    gaussian = jax.random.normal(key, shape=(batches, hilb.size))
    width = jnp.min(modulus) / (4.0 * hilb.n_particles)
    # The width gives the noise level. In the periodic case the
    # particles are evenly distributed between 0 and min(L). The
    # distance between the particles coordinates is therefore given by
    # min(L) / hilb.N. To avoid particles to have coincident
    # positions the noise level should be smaller than half this distance.
    # We choose width = min(L) / (4*hilb.N)
    noise = gaussian * width
    uniform = jnp.tile(jnp.linspace(0.0, jnp.min(modulus), hilb.size), (batches, 1))

    rs = jnp.where(jnp.equal(boundary, False), gaussian, (uniform + noise) % modulus)

    return jnp.asarray(rs, dtype=dtype)
