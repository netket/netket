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
import netket.experimental as nkx
import numpy as np
import optax
import pytest
from jax import numpy as jnp


graph_full = nk.graph.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

partial_model_pairs = [
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nk.models.ARNNConv1D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=2,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nk.models.FastARNNConv1D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=2,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="conv1d",
    ),
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nk.models.ARNNConv1D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=2,
                kernel_dilation=2,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nk.models.FastARNNConv1D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=2,
                kernel_dilation=2,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="conv1d_dilation",
    ),
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nk.models.ARNNConv2D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=(2, 3),
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nk.models.FastARNNConv2D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=(2, 3),
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="conv2d",
    ),
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nk.models.ARNNConv2D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=(2, 3),
                kernel_dilation=(2, 2),
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nk.models.FastARNNConv2D(
                hilbert=hilbert,
                layers=3,
                features=5,
                kernel_size=(2, 3),
                kernel_dilation=(2, 2),
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="conv2d_dilation",
    ),
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nkx.models.LSTMNet(
                hilbert=hilbert,
                layers=3,
                features=5,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nkx.models.FastLSTMNet(
                hilbert=hilbert,
                layers=3,
                features=5,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="lstm",
    ),
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nkx.models.LSTMNet(
                hilbert=hilbert,
                layers=3,
                features=5,
                graph=nk.graph.Square(2),
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nkx.models.FastLSTMNet(
                hilbert=hilbert,
                layers=3,
                features=5,
                graph=nk.graph.Square(2),
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="lstm_square",
    ),
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nkx.models.LSTMNet(
                hilbert=hilbert,
                layers=3,
                features=5,
                graph=graph_full,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nkx.models.FastLSTMNet(
                hilbert=hilbert,
                layers=3,
                features=5,
                graph=graph_full,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="lstm_full",
    ),
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nkx.models.GRUNet1D(
                hilbert=hilbert,
                layers=3,
                features=5,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nkx.models.FastGRUNet1D(
                hilbert=hilbert,
                layers=3,
                features=5,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="gru",
    ),
]


model_pairs = {param.id: param.values[0] for param in partial_model_pairs}

hilberts = {
    "spin_1/2": nk.hilbert.Spin(s=1 / 2, N=4),
    "spin_1": nk.hilbert.Spin(s=1, N=4),
    "fock": nk.hilbert.Fock(n_max=3, N=4),
}
param_dtypes = {"float64": jnp.float64, "complex128": jnp.complex128}


def _cases(ids):
    """
    Parametrize over `partial_model_pair, hilbert, param_dtype, machine_pow` from
    ids of the form "model-hilbert-dtype-machine_pow".

    The full product (8 models x 3 Hilbert spaces x 2 dtypes x 2 machine_pow) is
    too expensive, as every case is dominated by compilation. Instead every model
    runs once, and the Hilbert spaces, dtypes and machine_pow are spread over the
    models.
    """
    cases = []
    for id in ids:
        model, hilbert, dtype, machine_pow = id.split("-")
        cases.append(
            pytest.param(
                model_pairs[model],
                hilberts[hilbert],
                param_dtypes[dtype],
                int(machine_pow),
                id=id,
            )
        )
    return pytest.mark.parametrize(
        "partial_model_pair, hilbert, param_dtype, machine_pow", cases
    )


@_cases(
    [
        "conv1d-spin_1/2-complex128-1",
        "conv1d_dilation-spin_1-complex128-2",
        "conv2d-spin_1/2-float64-2",
        "conv2d_dilation-spin_1-float64-2",
        "lstm-spin_1/2-float64-1",
        "lstm_square-spin_1/2-float64-2",
        "lstm_full-fock-complex128-2",
        "gru-fock-float64-1",
    ]
)
def test_vmc_same(partial_model_pair, hilbert, param_dtype, machine_pow):
    model1 = partial_model_pair[0](hilbert, param_dtype, machine_pow)
    model2 = partial_model_pair[1](hilbert, param_dtype, machine_pow)

    sampler1 = nk.sampler.ARDirectSampler(hilbert)
    vstate1 = nk.vqs.MCState(sampler1, model1, n_samples=6, seed=123, sampler_seed=456)
    assert vstate1.n_discard_per_chain == 0
    samples1 = vstate1.sample()

    H = nk.operator.LocalOperator(hilbert)
    for i in range(hilbert.size):
        j = i % hilbert.size
        H += -nk.operator.spin.sigmax(hilbert, i)
        H += nk.operator.spin.sigmaz(hilbert, i) @ nk.operator.spin.sigmaz(hilbert, j)

    optimizer = optax.adam(learning_rate=1e-3)
    vmc1 = nk.driver.VMC(H, optimizer, variational_state=vstate1)
    vmc1.run(n_iter=3)
    samples_trained1 = vstate1.sample()

    sampler2 = nk.sampler.ARDirectSampler(hilbert)
    vstate2 = nk.vqs.MCState(sampler2, model2, n_samples=6, seed=123, sampler_seed=456)
    samples2 = vstate2.sample()

    # Samples from FastARNN* should be the same as those from ARNN*
    np.testing.assert_allclose(samples2, samples1)

    vmc2 = nk.driver.VMC(H, optimizer, variational_state=vstate2)
    vmc2.run(n_iter=3)
    samples_trained2 = vstate2.sample()

    # Samples from FastARNN* after training should be the same as those from ARNN*
    np.testing.assert_allclose(samples_trained2, samples_trained1)
