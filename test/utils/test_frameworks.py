import builtins

import pytest

import numpy as np

import jax
import jax.numpy as jnp

import netket as nk


@pytest.fixture
def block_import(monkeypatch, blocked_modules):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in blocked_modules:
            raise ImportError(f"Blocked module '{name}' was imported")
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


@pytest.mark.usefixtures("block_import")
@pytest.mark.parametrize("blocked_modules", [["haiku"]])
def test_jax_framework_works_without_haiku():
    def init(rng, in_shape):
        out_shape = (1,)
        pars = {"0": jnp.ones((1,))}
        return out_shape, pars

    def apply(pars, x, **_):
        return pars["0"] * jnp.sum(x)

    hi = nk.hilbert.Qubit(8)
    sampler = nk.sampler.MetropolisLocal(hi)
    nk.vqs.MCState(sampler, model=(init, apply))


def test_haiku_framework():
    pytest.importorskip("dm_haiku")
    import haiku as hk

    def apply(x):
        net = hk.Sequential(
            [
                hk.Linear(10, with_bias=False),
                jax.nn.relu,
                hk.Linear(1, with_bias=False),
            ]
        )
        return net(x)[..., 0]

    hi = nk.hilbert.Qubit(8)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model=hk.transform(apply))

    assert vstate.n_parameters == hi.size * 10 + 10

    logpsi = vstate.log_value(hi.all_states())
    assert logpsi.shape == (hi.n_states,)


def test_equinox_framework():
    pytest.importorskip("equinox")
    import equinox as eqx

    class SupportBatch(eqx.Module):
        submodule: eqx.Module

        def __init__(self, submodule):
            self.submodule = submodule

        def __call__(self, x, **kwargs):
            return jax.vmap(lambda x: self.submodule(x, **kwargs))(x)

    L = 8

    ma = SupportBatch(
        eqx.nn.MLP(
            in_size=L, out_size="scalar", width_size=8, depth=1, key=jax.random.key(1)
        )
    )

    hi = nk.hilbert.Qubit(L)
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, ma)

    assert vstate.n_parameters == hi.size * (ma.submodule.width_size + 2) + 1

    logpsi = vstate.log_value(hi.all_states())
    assert logpsi.shape == (hi.n_states,)

    np.testing.assert_allclose(vstate.model(hi.all_states()), logpsi)
