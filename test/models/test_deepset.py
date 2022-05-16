# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import jax
import jax.numpy as jnp

import netket as nk


@pytest.mark.parametrize(
    "cusp_exponent", [pytest.param(None, id="cusp=None"), pytest.param(5, id="cusp=5")]
)
@pytest.mark.parametrize(
    "L",
    [
        pytest.param(1.0, id="1D"),
        pytest.param((1.0, 1.0), id="2D-Square"),
        pytest.param((1.0, 0.5), id="2D-Rectangle"),
    ],
)
def test_deepsets(cusp_exponent, L):

    hilb = nk.hilbert.Particle(N=2, L=L, pbc=True)
    sdim = len(hilb.extent)
    x = jnp.hstack([jnp.ones(4), -jnp.ones(4)]).reshape(1, -1)
    xp = jnp.roll(x, sdim)
    ds = nk.models.DeepSetRelDistance(
        hilbert=hilb,
        cusp_exponent=cusp_exponent,
        layers_phi=2,
        layers_rho=2,
        features_phi=(10, 10),
        features_rho=(10, 1),
    )
    p = ds.init(jax.random.PRNGKey(42), x)

    assert jnp.allclose(ds.apply(p, x), ds.apply(p, xp))


def test_deepsets_error():
    hilb = nk.hilbert.Particle(N=2, L=1.0, pbc=True)
    sdim = len(hilb.extent)

    x = jnp.hstack([jnp.ones(4), -jnp.ones(4)]).reshape(1, -1)
    xp = jnp.roll(x, sdim)
    ds = nk.models.DeepSetRelDistance(
        hilbert=hilb,
        layers_phi=3,
        layers_rho=3,
        features_phi=(10, 10),
        features_rho=(10, 1),
    )
    with pytest.raises(ValueError):
        p = ds.init(jax.random.PRNGKey(42), x)

    with pytest.raises(AssertionError):
        ds = nk.models.DeepSetRelDistance(
            hilbert=hilb,
            layers_phi=2,
            layers_rho=2,
            features_phi=(10, 10),
            features_rho=(10, 2),
        )
        p = ds.init(jax.random.PRNGKey(42), x)

    with pytest.raises(ValueError):
        ds = nk.models.DeepSetRelDistance(
            hilbert=nk.hilbert.Particle(N=2, L=1.0, pbc=False),
            layers_phi=2,
            layers_rho=2,
            features_phi=(10, 10),
            features_rho=(10, 2),
        )
        p = ds.init(jax.random.PRNGKey(42), x)
