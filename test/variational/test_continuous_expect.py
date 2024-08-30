import jax.numpy as jnp
import numpy as np
import netket as nk

import flax.linen as nn


class test(nn.Module):
    @nn.compact
    def __call__(self, x):
        _ = self.param("nothing", lambda *args: jnp.ones(1))
        if len(x.shape) != 1:
            return jnp.array(x.size * [1.0])
        return 1.0


class test2(nn.Module):
    @nn.compact
    def __call__(self, x):
        nothing = self.param("nothing", lambda *args: jnp.ones(1))

        sol = jnp.sum(nothing**2 * x, axis=-1)
        return sol


# continuous preparations
def v1(x):
    return 1 / jnp.sqrt(2 * jnp.pi) * jnp.sum(jnp.exp(-0.5 * ((x - 2.5) ** 2)), axis=-1)


def v2(x):
    return 1 / jnp.sqrt(2 * jnp.pi) * jnp.sum(jnp.exp(-0.5 * ((x - 2.5) ** 2)), axis=-1)


def test_expect():
    hilb = nk.hilbert.Particle(N=1, L=5, pbc=True)
    pot = nk.operator.PotentialEnergy(hilb, v1)
    kin = nk.operator.KineticEnergy(hilb, mass=1.0)
    e = pot + kin
    sab = nk.sampler.MetropolisGaussian(hilb, sigma=1.0, n_chains=16, sweep_size=1)

    model = test()
    model2 = test2()
    vs_continuous = nk.vqs.MCState(
        sab,
        model,
        n_samples=256 * 1024,
        n_discard_per_chain=2048,
        sampler_seed=1234,
    )
    vs_continuous2 = nk.vqs.MCState(
        sab,
        model2,
        n_samples=1024 * 1024,
        n_discard_per_chain=2048,
        sampler_seed=1234,
    )

    assert vs_continuous.chunk_size is None
    assert vs_continuous2.chunk_size is None
    # x = vs_continuous2.samples.reshape(-1, 1)
    sol_nc = vs_continuous.expect(pot)
    O_stat_nc, O_grad_nc = vs_continuous2.expect_and_grad(e)
    O_grad_nc, _ = nk.jax.tree_ravel(O_grad_nc)

    # O_grad_exact = 2 * jnp.dot(x.T, (v1(x) - jnp.mean(v1(x), axis=0))) / x.shape[0]
    r"""
    :math:`<V> = \int_0^5 dx V(x) |\psi(x)|^2 / \int_0^5 |\psi(x)|^2 = 0.1975164 (\psi = 1)`
    :math:`<\nabla V> = \nabla_p \int_0^5 dx V(x) |\psi(x)|^2 / \int_0^5 |\psi(x)|^2 = -0.140256 (\psi = \exp(p^2 x))`
    """
    np.testing.assert_allclose(0.1975164, sol_nc.mean, atol=1.5e-3)
    np.testing.assert_allclose(-0.140256, O_grad_nc, atol=1.5e-3)

    vs_continuous.chunk_size = 128
    vs_continuous2.chunk_size = 128

    assert vs_continuous.chunk_size == 128
    assert vs_continuous2.chunk_size == 128

    sol = vs_continuous.expect(pot)
    O_stat, O_grad = vs_continuous2.expect_and_grad(e)
    O_grad, _ = nk.jax.tree_ravel(O_grad)

    np.testing.assert_allclose(sol_nc.mean, sol.mean, atol=1e-8)
    np.testing.assert_allclose(O_grad_nc, O_grad, atol=1e-8)
