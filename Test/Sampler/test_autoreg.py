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
from jax import numpy as jnp


def test_ARSampler():
    batch = 3
    size = 4

    graph = nk.graph.Hypercube(length=size, n_dim=1, pbc=True)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)

    model = nk.models.ARNNDense(layers=3, features=5)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((batch, size)))

    sampler = nk.sampler.ARSampler(hilbert, n_chains=batch)

    sampler_state = sampler.init_state(model, params)
    samples, sampler_state = sampler.sample(model, params, chain_length=3)

    all_states = hilbert.all_states()
    for sample in samples:
        assert sample.shape == (sampler.n_chains, hilbert.size)
        for v in sample:
            assert v in all_states
