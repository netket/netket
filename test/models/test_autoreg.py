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
import netket.experimental as nkx
import numpy as np
import pytest
from jax import numpy as jnp
from netket.utils import HashableArray

from .. import common


@pytest.fixture
def skip(request):
    rate = request.config.getoption("--arnn_test_rate")
    rng = np.random.default_rng(common.hash_for_seed(request.node.callspec.id))
    if rng.random() > rate:
        pytest.skip(
            "Running only a portion of the tests for ARNN. Use --arnn_test_rate=1 to run all tests."
        )


graph_full = nk.graph.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])

partial_model_pairs = [
    pytest.param(
        (
            lambda hilbert, param_dtype, machine_pow: nk.models.ARNNDense(
                hilbert=hilbert,
                layers=3,
                features=5,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
            lambda hilbert, param_dtype, machine_pow: nk.models.FastARNNDense(
                hilbert=hilbert,
                layers=3,
                features=5,
                param_dtype=param_dtype,
                machine_pow=machine_pow,
            ),
        ),
        id="dense",
    ),
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

partial_models = [
    pytest.param(param.values[0][0], id=param.id) for param in partial_model_pairs
]
partial_models += [
    pytest.param(param.values[0][1], id="fast_" + param.id)
    for param in partial_model_pairs
]


@pytest.mark.parametrize("machine_pow", [1, 2])
@pytest.mark.parametrize("param_dtype", [jnp.float64, jnp.complex128])
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
class TestARNN:
    @pytest.mark.parametrize("partial_model", partial_models)
    def test_norm_autoreg(self, partial_model, hilbert, param_dtype, machine_pow, skip):
        batch_size = 3

        model = partial_model(hilbert, param_dtype, machine_pow)

        key_spins, key_model = jax.random.split(jax.random.PRNGKey(0))
        spins = hilbert.random_state(key_spins, size=batch_size)
        p, variables = model.init_with_output(
            key_model, spins, method=model.conditionals
        )

        @jax.jit
        def conditionals(inputs):
            return model.apply(variables, inputs, method=model.conditionals)

        @jax.jit
        def reorder(inputs):
            return model.apply(variables, inputs, axis=1, method=model.reorder)

        @jax.jit
        def inverse_reorder(inputs):
            return model.apply(variables, inputs, axis=1, method=model.inverse_reorder)

        p = reorder(p)

        # Test if the model is normalized
        # The result may not be very accurate, because it is in exp space
        psi = nk.nn.to_array(hilbert, model.apply, variables, normalize=False)
        assert (jnp.abs(psi) ** machine_pow).sum() == pytest.approx(
            1, rel=1e-5, abs=1e-5
        )

        # Test if the model is autoregressive
        for i in range(batch_size):
            for j in range(hilbert.size):
                # Change one input element at a time
                spins_new = reorder(spins)
                spins_new = spins_new.at[i, j].multiply(-1)
                spins_new = inverse_reorder(spins_new)

                p_new = conditionals(spins_new)
                p_new = reorder(p_new)

                # Sites after j can change, so we reset them before comparison
                p_new = p_new.at[i, j + 1 :].set(p[i, j + 1 :])

                np.testing.assert_allclose(p_new, p, err_msg=f"i={i} j={j}")

    @pytest.mark.parametrize("partial_model_pair", partial_model_pairs)
    def test_same(self, partial_model_pair, hilbert, param_dtype, machine_pow, skip):
        batch_size = 3

        model1 = partial_model_pair[0](hilbert, param_dtype, machine_pow)
        model2 = partial_model_pair[1](hilbert, param_dtype, machine_pow)

        key_spins, key_model = jax.random.split(jax.random.PRNGKey(0))
        spins = hilbert.random_state(key_spins, size=batch_size)
        variables = model2.init(key_model, spins, 0, method=model2.conditional)

        p1 = model1.apply(variables, spins, method=model1.conditionals)
        p2 = model2.apply(variables, spins, method=model2.conditionals)

        # Results from `FastARNN*.conditionals` should be the same as those from `ARNN*.conditionals`
        np.testing.assert_allclose(p2, p1)

        p3 = jnp.zeros_like(p1)
        params = variables["params"]
        cache = variables["cache"]
        indices = jnp.arange(hilbert.size)
        indices = model2.apply(variables, indices, method=model2.reorder)
        for i in indices:
            variables = {"params": params, "cache": cache}
            p_i, mutables = model2.apply(
                variables,
                spins,
                i,
                method=model2.conditional,
                mutable=["cache"],
            )
            cache = mutables["cache"]
            p3 = p3.at[:, i, :].set(p_i)

        # Results from `FastARNN*.conditional` should be the same as those from `ARNN*.conditionals`
        np.testing.assert_allclose(p3, p1)


def test_throwing():
    def build_model(hilbert):
        nk.models.ARNNConv1D(
            hilbert=hilbert,
            machine_pow=2,
            layers=3,
            features=5,
            kernel_size=2,
        )

    # Only homogeneous Hilbert spaces are supported
    with pytest.raises(ValueError):
        hilbert = nk.hilbert.Spin(s=1 / 2, N=4)
        hilbert = nk.hilbert.DoubledHilbert(hilbert)
        build_model(hilbert)


@pytest.mark.parametrize(
    "graph",
    [
        pytest.param(
            nk.graph.Grid(extent=[4, 4], pbc=False),
            id="4x4_obc",
        ),
        pytest.param(
            nk.graph.Grid(extent=[3, 5], pbc=False),
            id="3x5_obc",
        ),
        pytest.param(
            nk.graph.Grid(extent=[5, 3], pbc=False),
            id="5x3_obc",
        ),
    ],
)
def test_reorder_idx(graph):
    from netket.experimental.nn.rnn.ordering import (
        _get_inv_idx,
        _get_inv_reorder_idx,
        _get_prev_neighbors,
        get_snake_inv_reorder_idx,
        _get_snake_prev_neighbors,
    )

    inv_reorder_idx_1 = _get_inv_reorder_idx(graph)
    inv_reorder_idx_2 = get_snake_inv_reorder_idx(graph)
    assert inv_reorder_idx_1 == inv_reorder_idx_2

    reorder_idx_1 = _get_inv_idx(inv_reorder_idx_1)
    prev_neighbors_1 = _get_prev_neighbors(graph, reorder_idx_1)
    prev_neighbors_2 = _get_snake_prev_neighbors(graph)
    assert prev_neighbors_1 == prev_neighbors_2


def test_construct_rnn():
    def build_model(
        reorder_idx=None, inv_reorder_idx=None, prev_neighbors=None, graph=None
    ):
        model = nkx.models.LSTMNet(
            hilbert=nk.hilbert.Spin(s=1 / 2, N=4),
            layers=3,
            features=5,
            reorder_idx=reorder_idx,
            inv_reorder_idx=inv_reorder_idx,
            prev_neighbors=prev_neighbors,
            graph=graph,
        )

        # Call `setup` to check RNN layers
        inputs = jnp.zeros(4)
        model.init(nk.jax.PRNGKey(), inputs)

    reorder_idx = HashableArray(np.array([0, 1, 3, 2]))
    inv_reorder_idx = HashableArray(np.array([0, 1, 3, 2]))
    prev_neighbors = HashableArray(np.array([[-1, -1], [0, -1], [0, 3], [1, -1]]))
    graph = nk.graph.Square(2)

    build_model()
    build_model(graph=graph)
    build_model(reorder_idx=reorder_idx, graph=graph)
    build_model(inv_reorder_idx=inv_reorder_idx, graph=graph)
    build_model(
        reorder_idx=reorder_idx,
        inv_reorder_idx=inv_reorder_idx,
        prev_neighbors=prev_neighbors,
    )

    # When `prev_neighbors` is provided, you must also provide either `reorder_idx` or `inv_reorder_idx`
    with pytest.raises(ValueError):
        build_model(prev_neighbors=prev_neighbors)

    # When `reorder_idx` is provided, you must also provide either `prev_neighbors` or `graph`
    with pytest.raises(ValueError):
        build_model(reorder_idx=reorder_idx)

    # `inv_reorder_idx` is not the inverse of `reorder_idx`
    with pytest.raises(ValueError):
        build_model(
            reorder_idx=HashableArray(np.array([0, 1, 2, 3])),
            inv_reorder_idx=inv_reorder_idx,
            prev_neighbors=prev_neighbors,
        )

    # Site 1 is not a previous neighbor of site 0
    with pytest.raises(ValueError):
        build_model(
            reorder_idx=reorder_idx,
            inv_reorder_idx=inv_reorder_idx,
            prev_neighbors=HashableArray(np.array([[1, -1], [0, -1], [0, 3], [1, -1]])),
        )
