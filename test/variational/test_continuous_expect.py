import jax.numpy as jnp
import numpy as np
import netket as nk

# continuous preparations
def v1(x):
    return 1 / jnp.sqrt(2 * jnp.pi) * jnp.sum(jnp.exp(-0.5 * ((x - 2.5) ** 2)))


import flax.linen as nn


class test(nn.Module):
    @nn.compact
    def __call__(self, x):
        nothing = self.param("nothing", lambda *args: jnp.ones(1))
        if len(x.shape) != 1:
            return jnp.array(x.size * [1.0])
        return 1.0


hilb = nk.hilbert.Particle(N=1, L=5, pbc=True)
pot = nk.operator.PotentialEnergy(hilb, v1)
sab = nk.sampler.MetropolisGaussian(hilb, sigma=1.0, n_chains=16, n_sweeps=1)

model = test()
vs_continuous = nk.vqs.MCState(sab, model, n_samples=10 ** 6, n_discard=2000)


def test_expect():
    sol = vs_continuous.expect(pot)
    np.testing.assert_allclose(0.1975164, sol.mean, atol=10 ** (-3))
