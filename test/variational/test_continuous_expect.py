import jax.numpy as jnp
import netket as nk
from jax.nn.initializers import normal

# continuous preparations
def v1(x):
    return 1 / jnp.sqrt(2 * jnp.pi) * jnp.sum(jnp.exp(-0.5 * ((x - 2.0) ** 2)))


hilb = nk.hilbert.Particle(N=1, L=4, pbc=True)
pot = nk.operator.PotentialEnergy(hilb, v1)
sab = nk.sampler.MetropolisGaussian(hilb, sigma=1.0, n_chains=16, n_sweeps=1)

model = nk.models.Gaussian()
vs_continuous = nk.vqs.MCState(sab, model, n_samples=10 ** 6, n_discard=2000)


def test_expect():
    sol = vs_continuous.expect(pot)
    assert jnp.allclose(1.0, sol)
