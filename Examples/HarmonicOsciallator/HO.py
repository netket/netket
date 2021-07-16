import netket as nk
import jax.numpy as jnp

L = 10.0

hib = nk.hilbert.ContinuousBoson(N=5, L=(jnp.inf,), pbc=(False,))
hif = nk.hilbert.ContinuousFermion(
    N=5,
    L=(L, L, L),
    pbc=(True, True, True),
)

sab = nk.sampler.MetropolisGaussian(hib, sigma=1.0, n_chains=16, n_sweeps=1)
saf = nk.sampler.MetropolisGaussian(hif, sigma=1.0, n_chains=16, n_sweeps=1)

print(sab.sample)
print(saf.sample)
