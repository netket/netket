import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import netket as nk
import netket.experimental as nkx

import flax.linen as nn


class test(nn.Module):
    @nn.compact
    def __call__(self, x):
        _ = self.param("nothing", lambda *args: jnp.ones(1))
        return jnp.ones_like(x[..., 0])


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
    hilb = nkx.hilbert.Particle(N=1, geometry=nkx.geometry.Cell(d=1, L=5.0, pbc=True))
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
    O_grad_nc, _ = ravel_pytree(O_grad_nc)

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
    O_grad, _ = ravel_pytree(O_grad)

    np.testing.assert_allclose(sol_nc.mean, sol.mean, atol=1e-7)
    np.testing.assert_allclose(O_grad_nc, O_grad, atol=1e-7)


class GatherModel(nn.Module):
    """Model that gathers a parameter by an index array.

    The gather op (advanced indexing) on a parameter leaf is what triggers the
    regression in `test_continuous_kinetic_param_gather_chunked`: under sharding
    the parameter must be `pvary`/`pcast`-ed into the shard_map's Manual mesh,
    otherwise `jnp.take`/indexing raises a mesh-mismatch error.
    """

    @nn.compact
    def __call__(self, x):
        coeff = self.param("coeff", lambda k, *a: jnp.linspace(0.1, 0.4, 4))
        vecs = self.param("vecs", lambda k, *a: jnp.ones((6, x.shape[-1])))
        idx = np.array([0, 1, 2, 3, 0, 1])
        c = coeff[idx]  # gather a parameter leaf by an index array
        return jnp.einsum(
            "k,...k->...", c, jnp.cos(jnp.einsum("...d,kd->...k", x, vecs))
        )


def test_continuous_kinetic_param_gather_chunked():
    """Regression test for a sharding bug in the chunked continuous-operator kernel.

    `KineticEnergy` differentiates `logpsi(params, x)`, so it actually evaluates
    the model parameters (unlike `PotentialEnergy`). The chunked continuous kernel
    used to capture `pars` in a closure instead of passing it as an explicit
    argument of `nkjax.apply_chunked`, so `pars` never crossed the shard_map
    boundary and kept its (Auto-mesh) sharding. Most ops tolerate this, but a
    gather of a parameter leaf raised a Manual-vs-Auto mesh-mismatch error under
    sharding. This test runs the chunked path and checks it matches the unchunked
    result. Under the distributed/sharded test runs it guards against regressions.
    """
    hilb = nkx.hilbert.Particle(
        N=1, geometry=nkx.geometry.Cell(d=2, L=(5.0,) * 2, pbc=True)
    )
    kin = nk.operator.KineticEnergy(hilb, mass=1.0)
    sab = nk.sampler.MetropolisGaussian(hilb, sigma=1.0, n_chains=16, sweep_size=1)

    vs = nk.vqs.MCState(sab, GatherModel(), n_samples=512, sampler_seed=1234, seed=1234)

    assert vs.chunk_size is None
    sol_nc = vs.expect(kin)

    vs.chunk_size = 8
    assert vs.chunk_size == 8
    sol = vs.expect(kin)  # used to raise under sharding

    np.testing.assert_allclose(sol_nc.mean, sol.mean, atol=1e-7)
