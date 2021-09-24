import netket as nk

import jax.numpy as jnp

import netket.operator


def v(x):
    return 0.5 * jnp.linalg.norm(x) ** 2


L = 10.0

hib = nk.hilbert.ContinuousBoson(N=5, L=(jnp.inf,), pbc=(False,))

sab = nk.sampler.MetropolisGaussian(hib, sigma=1.0, n_chains=16, n_sweeps=1)
model = nk.models.Gaussian(dtype=float)

ha = netket.operator.KineticPotential(hib, v)

vs = nk.vqs.MCState(sab, model, n_samples=10 ** 5, n_discard=2000)

op = nk.optimizer.Sgd(0.01)
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational monte carlo driver
gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)
# Run the optimization for 300 iterations
gs.run(n_iter=300, out="HO_5particles")
