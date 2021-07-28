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

import netket as nk
import numpy as np
import optax
import pytest
from jax import numpy as jnp

from .. import common

pytestmark = common.skipif_mpi


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
@pytest.mark.parametrize(
    "hilbert",
    [
        pytest.param(
            nk.hilbert.Spin(s=1 / 2, N=4),
            id="spin_1/2",
        ),
        pytest.param(
            nk.hilbert.Spin(s=1, N=4),
            id="spin_1",
        ),
        pytest.param(
            nk.hilbert.Fock(n_max=3, N=4),
            id="fock",
        ),
    ],
)
@pytest.mark.parametrize(
    "partial_models",
    [
        # pytest.param(
        #     (
        #         lambda hilbert, dtype: nk.models.ARNNDense(
        #             hilbert=hilbert,
        #             layers=3,
        #             features=5,
        #             dtype=dtype,
        #         ),
        #         lambda hilbert, dtype: nk.models.FastARNNDense(
        #             hilbert=hilbert,
        #             layers=3,
        #             features=5,
        #             dtype=dtype,
        #         ),
        #     ),
        #     id="dense",
        # ),
        pytest.param(
            (
                lambda hilbert, dtype: nk.models.ARNNConv1D(
                    hilbert=hilbert,
                    layers=3,
                    features=5,
                    kernel_size=2,
                    dtype=dtype,
                ),
                lambda hilbert, dtype: nk.models.FastARNNConv1D(
                    hilbert=hilbert,
                    layers=3,
                    features=5,
                    kernel_size=2,
                    dtype=dtype,
                ),
            ),
            id="conv1d",
        ),
        pytest.param(
            (
                lambda hilbert, dtype: nk.models.ARNNConv1D(
                    hilbert=hilbert,
                    layers=3,
                    features=5,
                    kernel_size=2,
                    kernel_dilation=2,
                    dtype=dtype,
                ),
                lambda hilbert, dtype: nk.models.FastARNNConv1D(
                    hilbert=hilbert,
                    layers=3,
                    features=5,
                    kernel_size=2,
                    kernel_dilation=2,
                    dtype=dtype,
                ),
            ),
            id="conv1d_dilation",
        ),
        pytest.param(
            (
                lambda hilbert, dtype: nk.models.ARNNConv2D(
                    hilbert=hilbert,
                    layers=3,
                    features=5,
                    kernel_size=(2, 3),
                    dtype=dtype,
                ),
                lambda hilbert, dtype: nk.models.FastARNNConv2D(
                    hilbert=hilbert,
                    layers=3,
                    features=5,
                    kernel_size=(2, 3),
                    dtype=dtype,
                ),
            ),
            id="conv2d",
        ),
        pytest.param(
            (
                lambda hilbert, dtype: nk.models.ARNNConv2D(
                    hilbert=hilbert,
                    layers=3,
                    features=5,
                    kernel_size=(2, 3),
                    kernel_dilation=(2, 2),
                    dtype=dtype,
                ),
                lambda hilbert, dtype: nk.models.FastARNNConv2D(
                    hilbert=hilbert,
                    layers=3,
                    features=5,
                    kernel_size=(2, 3),
                    kernel_dilation=(2, 2),
                    dtype=dtype,
                ),
            ),
            id="conv2d_dilation",
        ),
    ],
)
def test_AR_VMC(partial_models, hilbert, dtype):
    model1 = partial_models[0](hilbert, dtype)
    model2 = partial_models[1](hilbert, dtype)

    sampler1 = nk.sampler.ARDirectSampler(hilbert, n_chains=3)
    vstate1 = nk.vqs.MCState(sampler1, model1, n_samples=6, seed=123, sampler_seed=456)
    assert vstate1.n_discard_per_chain == 0
    samples1 = vstate1.sample()

    graph = nk.graph.Hypercube(length=hilbert.size, n_dim=1)
    H = nk.operator.Ising(hilbert=hilbert, graph=graph, h=1)
    optimizer = optax.adam(learning_rate=1e-3)
    vmc1 = nk.VMC(H, optimizer, variational_state=vstate1)
    vmc1.run(n_iter=3)
    samples_trained1 = vstate1.sample()

    sampler2 = nk.sampler.ARDirectSampler(hilbert, n_chains=3)
    vstate2 = nk.vqs.MCState(sampler2, model2, n_samples=6, seed=123, sampler_seed=456)
    samples2 = vstate2.sample()

    # Samples from FastARNN* should be the same as those from ARNN*
    np.testing.assert_allclose(samples2, samples1)

    vmc2 = nk.VMC(H, optimizer, variational_state=vstate2)
    vmc2.run(n_iter=3)
    samples_trained2 = vstate2.sample()

    # Samples from FastARNN* after training should be the same as those from ARNN*
    np.testing.assert_allclose(samples_trained2, samples_trained1)
