import netket as nk

L = 10.

hi = nk.hilbert.Particles(N=5, L = (L,), pbc = (True,), ptype='boson')

sa = nk.sampler.MetropolisGaussian(hi, sigma=0.5, n_chains=10, n_sweeps=1)

print(sa.sample)