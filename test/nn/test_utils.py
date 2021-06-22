# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import jax.numpy as jnp
import numpy as np

import netket as nk

from .. import common  # noqa: F401

SEED = 111


@pytest.fixture(
    params=[pytest.param(M, id=f"Fock(M={M})") for M in [0, 2, 3, 4, 5, 6, 8]]
)
def vstate(request):
    M = request.param
    # keep this a prime number so we get different sizes on every rank...
    hi = nk.hilbert.Fock(M, 1)

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


@pytest.fixture(
    params=[pytest.param(M, id=f"Fock(M={M})") for M in [0, 2, 3, 4, 5, 6, 8]]
)
def vstate_rho(request):
    M = request.param
    # keep this a prime number so we get different sizes on every rank...
    hi = nk.hilbert.Fock(M, 1)

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
