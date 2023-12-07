# Copyright 2022 The NetKet Authors - All rights reserved.
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
import numpy as np
import jax
import jax.numpy as jnp

import netket as nk
import netket.nn as nknn


def test_mlp_alpha():
    ma = nk.nn.blocks.MLP(hidden_dims_alpha=(4, 5))
    x = np.zeros((16,))
    pars = ma.init(nk.jax.PRNGKey(), x)
    assert pars["params"]["Dense_0"]["kernel"].shape == (16, 4 * 16)
    assert pars["params"]["Dense_1"]["kernel"].shape == (4 * 16, 5 * 16)


def test_mlp_dimensions():
    ma = nk.nn.blocks.MLP(
        output_dim=3,
        hidden_dims=(16, 32),
        param_dtype=np.float64,
        hidden_activations=None,
        output_activation=nk.nn.gelu,
        use_output_bias=True,
    )
    x = np.zeros((1024, 16))

    pars = ma.init(nk.jax.PRNGKey(), x)
    out = ma.apply(pars, x)
    assert out.shape[-1] == 3

    ma = nk.nn.blocks.MLP(output_dim=1, hidden_dims=(16, 32))
    pars = ma.init(nk.jax.PRNGKey(), x)
    out = ma.apply(pars, x)
    assert out.shape[-1] == 1

    ma = nk.nn.blocks.MLP(output_dim=3, hidden_dims=None)
    pars = ma.init(nk.jax.PRNGKey(), x)
    assert pars["params"]["Dense_0"]["kernel"].shape == (x.shape[-1], 3)
    assert "Dense_1" not in pars["params"]


def _eval_model(ma):
    # test input, only throws errors when init or apply is called
    x = np.zeros((1024, 16))
    pars = ma.init(nk.jax.PRNGKey(), x)
    _ = ma.apply(pars, x)


def test_mlp_input():
    # raise because different length
    with pytest.raises(ValueError):
        ma = nk.nn.blocks.MLP(
            output_dim=1,
            hidden_dims=(16, 33),
            hidden_activations=[nk.nn.gelu, nk.nn.gelu, nk.nn.gelu],
        )
        _eval_model(ma)

    # this must run
    ma = nk.nn.blocks.MLP(
        output_dim=1, hidden_dims=(16, 32), hidden_activations=[nk.nn.gelu, nk.nn.gelu]
    )
    _eval_model(ma)

    ma = nk.nn.blocks.MLP(output_dim=1, hidden_dims_alpha=(1, 2))
    _eval_model(ma)

    ma = nk.nn.blocks.MLP(output_dim=1)
    _eval_model(ma)

    with pytest.raises(ValueError):
        # raise because different length
        ma = nk.nn.blocks.MLP(
            output_dim=1,
            hidden_dims=(16, 32),
            hidden_activations=[nk.nn.gelu, nk.nn.gelu, nk.nn.gelu],
        )
        _eval_model(ma)

    with pytest.raises(ValueError):
        # cannot be specified together
        ma = nk.nn.blocks.MLP(
            output_dim=1, hidden_dims=(16, 32), hidden_dims_alpha=(1, 1)
        )
        _eval_model(ma)


def test_deepset():
    """Test the permutation invariance"""
    L = (1.0, 1.0)
    n_particles = 6
    hilb = nk.hilbert.Particle(N=n_particles, L=L, pbc=True)
    sdim = len(hilb.extent)
    key = jax.random.PRNGKey(42)
    x = hilb.random_state(key, size=1024)
    x = x.reshape(x.shape[0], n_particles, sdim)

    xp = jnp.roll(x, 2, axis=-2)  # permute the particles

    ds = nk.nn.blocks.DeepSetMLP(features_phi=(16, 16), features_rho=(16, 1))
    params = ds.init(key, x)
    out = ds.apply(params, x)
    outp = ds.apply(params, xp)
    assert out.shape == outp.shape
    assert out.shape[-1] == 1

    np.testing.assert_allclose(out, outp)

    ds = nk.nn.blocks.DeepSetMLP(
        features_phi=16, features_rho=32, output_activation=nknn.gelu
    )
    params = ds.init(key, x)
    out = ds.apply(params, x)
    assert params["params"]["ds_phi"]["Dense_0"]["kernel"].shape == (x.shape[-1], 16)
    assert params["params"]["ds_rho"]["Dense_0"]["kernel"].shape == (16, 32)

    ds = nk.nn.blocks.DeepSetMLP(features_phi=(16,), features_rho=(16, 3))
    params = ds.init(key, x)
    out = ds.apply(params, x)
    assert out.shape[-1] == 3

    # flexible, should still work
    ds = nk.nn.blocks.DeepSetMLP(
        features_phi=None,
        features_rho=None,
        output_activation=None,
        hidden_activation=None,
        pooling=jnp.prod,
        param_dtype=complex,
    )
    params = ds.init(key, x)
    out = ds.apply(params, x)
