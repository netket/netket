import netket as nk
import jax.numpy as jnp

L = 10.0

hilb = nk.hilbert.Particle(N=5, L=(jnp.inf,), pbc=(False,))

sab = nk.sampler.MetropolisGaussian(hib, sigma=1.0, n_chains=16, n_sweeps=1)
saf = nk.sampler.MetropolisGaussian(hif, sigma=1.0, n_chains=16, n_sweeps=1)

print(sab.sample)
print(saf.sample)
