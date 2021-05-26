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

import jax
import netket as nk
import pytest
from jax import numpy as jnp


@pytest.mark.parametrize("s", [1 / 2, 1])
def test_ARDirectSampler(s):
    L = 4
    n_chains = 3

    hilbert = nk.hilbert.Spin(s=s, N=L)
    model = nk.models.ARNNDense(hilbert=hilbert, layers=3, features=5)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((n_chains, L)))

    sampler = nk.sampler.ARDirectSampler(hilbert, n_chains=n_chains)
    samples, _ = sampler.sample(model, params, chain_length=3)

    all_states = hilbert.all_states()
    for sample in samples:
        assert sample.shape == (sampler.n_chains, L)
        for v in sample:
            assert v in all_states
