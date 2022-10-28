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

import netket as nk


def test_mlp_model_output():
    # make sure that the output of a model is flattened
    ma = nk.models.MLP(
        hidden_dims=(16, 32),
        param_dtype=np.float64,
        hidden_activations=None,
        output_activation=nk.nn.gelu,
        use_output_bias=True,
    )
    x = np.zeros((2, 1024, 16))

    pars = ma.init(nk.jax.PRNGKey(), x)
    out = ma.apply(pars, x)
    assert out.shape == (2, 1024)

    ma = nk.models.MLP(
        hidden_dims=None,
        param_dtype=np.complex128,
        hidden_activations=None,
        output_activation=None,
        use_output_bias=False,
    )
    x = np.zeros((2, 1024, 16))

    pars = ma.init(nk.jax.PRNGKey(), x)
    out = ma.apply(pars, x)
    assert out.shape == (2, 1024)
