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

from collections.abc import Iterable

from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.graph import AbstractGraph
from netket.models.autoreg import _get_feature_list
from netket.models.fast_autoreg import FastARNNSequential
from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc

from netket.experimental.models.rnn import RNN
from netket.experimental.nn.rnn import (
    FastRNNLayer,
    GRU1DCell,
    LSTMCell,
    default_kernel_init,
    ensure_prev_neighbors,
)


class FastRNN(FastARNNSequential):
    """
    Base class for recurrent neural networks with fast sampling.

    The fast autoregressive sampling is described in `Ramachandran et. {\\it al} <https://arxiv.org/abs/1704.06001>`_.
    To generate one sample using an autoregressive network, we need to evaluate the network `N` times, where `N` is
    the number of input sites. But actually we only change one input site each time, and not all intermediate results
    depend on the changed input because of the autoregressive property, so we can cache unchanged intermediate results
    and avoid repeated computation.

    This optimization is particularly useful for RNN where each output site of a layer only depends on a small number of
    input sites. In the slow RNN, we need to run `N` RNN steps in each layer during each AR sampling step. While in the
    fast RNN, we cache the relevant hidden memories in each layer from the previous AR sampling step, and only run one
    RNN step to update from the changed input.

    See :class:`netket.experimental.models.RNN` for explanation of the arguments related to
    the autoregressive order.
    """

    layers: int
    """number of layers."""
    features: Iterable[int] | int
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    reorder_idx: HashableArray | None = None
    """indices to transform the inputs from unordered to ordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    inv_reorder_idx: HashableArray | None = None
    """indices to transform the inputs from ordered to unordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    prev_neighbors: HashableArray | None = None
    """previous neighbors of each site."""
    graph: AbstractGraph | None = None
    """graph of the physical system."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def __post_init__(self):
        reorder_idx, inv_reorder_idx, prev_neighbors = ensure_prev_neighbors(
            reorder_idx=self.reorder_idx,
            inv_reorder_idx=self.inv_reorder_idx,
            prev_neighbors=self.prev_neighbors,
            graph=self.graph,
        )

        self.reorder_idx = reorder_idx
        self.inv_reorder_idx = inv_reorder_idx
        self.prev_neighbors = prev_neighbors
        super().__post_init__()

    def reorder(self, inputs: Array, axis: int = 0) -> Array:
        return RNN.reorder(self, inputs, axis)

    def inverse_reorder(self, inputs: Array, axis: int = 0) -> Array:
        return RNN.inverse_reorder(self, inputs, axis)

    def _take_prev_site(self, inputs: Array, index: int) -> Array:
        if self.reorder_idx is None:
            k = index
            prev_index = k - 1
        else:
            k = jnp.asarray(self.inv_reorder_idx)[index]
            prev_index = jnp.asarray(self.reorder_idx)[k - 1]

        inputs_i = inputs[:, prev_index, :]
        inputs_i = jnp.where(k == 0, 0, inputs_i)
        return inputs_i


class FastLSTMNet(FastRNN):
    """
    Long short-term memory network with fast sampling.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastRNNLayer(
                cell=LSTMCell(
                    features=features[i],
                    param_dtype=self.param_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
                size=self.hilbert.size,
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
            )
            for i in range(self.layers)
        ]


class FastGRUNet1D(FastRNN):
    """
    Gated recurrent unit network with fast sampling. Only supports one previous neighbor at each site.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastRNNLayer(
                cell=GRU1DCell(
                    features=features[i],
                    param_dtype=self.param_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
                size=self.hilbert.size,
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
            )
            for i in range(self.layers)
        ]
