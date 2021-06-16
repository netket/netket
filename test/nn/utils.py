import pytest

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
from functools import partial
import itertools

import tarfile
import glob
from io import BytesIO

from flax import serialization

import netket as nk

from .. import common

SEED = 111


@pytest.fixture(params=[pytest.param(M, id=f"Fock(M={M})") for M in [0, 3, 5, 7, 9]])
def vstate(request):
    M = request.param
    # keep this a prime number so we get different sizes on every rank...
    hi = nk.hilbert.Fock(7, 1)

    ma = nk.models.RBM(
        alpha=1,
        dtype=float,
        hidden_bias_init=nk.nn.initializers.normal(),
        visible_bias_init=nk.nn.initializers.normal(),
    )

    return nk.variational.MCState(
        nk.sampler.MetropolisLocal(hi),
        ma,
    )


@pytest.mark.parametrize("normalize", [True, False])
def test_to_array(vstate, normalize):
    psi = vstate.to_array(normalize=normalize)

    if normalize:
        np.testing.assert_allclose(jnp.linalg.norm(psi), 1.0)

    psi_norm = psi / jnp.linalg.norm(psi)

    assert psi.shape == (vstate.hilbert.n_states,)

    x = vstate.hilbert.all_states()
    psi_exact = jnp.exp(vstate.log_value(x))
    psi_exact = psi_exact / jnp.linalg.norm(psi_exact)

    np.testing.assert_allclose(psi_norm, psi_exact)


@pytest.fixture(params=[pytest.param(M, id=f"Fock(M={M})") for M in [0, 5, 9]])
def vstate_rho(request):
    M = request.param
    # keep this a prime number so we get different sizes on every rank...
    hi = nk.hilbert.Fock(7, 1)

    ma = nk.models.NDM()

    return nk.variational.MCMixedState(
        nk.sampler.MetropolisLocal(nk.hilbert.DoubledHilbert(hi)),
        ma,
    )


@pytest.mark.parametrize("normalize", [True, False])
def test_to_matrix(vstate_rho, normalize):
    rho = vstate_rho.to_matrix(normalize=normalize)

    if normalize:
        np.testing.assert_allclose(jnp.trace(rho), 1.0)

    rho_norm = rho / jnp.trace(rho)

    assert rho.shape == (
        vstate_rho.hilbert.physical.n_states,
        vstate_rho.hilbert.physical.n_states,
    )

    x = vstate_rho.hilbert.all_states()
    rho_exact = jnp.exp(vstate_rho.log_value(x)).reshape(rho.shape)
    rho_exact = rho_exact / jnp.trace(rho_exact)

    np.testing.assert_allclose(rho_norm, rho_exact)
