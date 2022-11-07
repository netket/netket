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

from typing import Iterable, Optional, Union

from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.graph import AbstractGraph
from netket.models.autoreg import _get_feature_list
from netket.models.fast_autoreg import FastARNNSequential
from netket.models.rnn import RNN, _ensure_prev_neighbors
from netket.nn.fast_rnn import FastGRULayer1D, FastLSTMLayer1D
from netket.nn.fast_rnn_2d import FastLSTMLayer2D
from netket.nn.rnn import default_kernel_init
from netket.utils import deprecate_dtype, HashableArray
from netket.utils.types import Array, DType, NNInitFunc


class FastRNN(FastARNNSequential):
    """
    Base class for recurrent neural networks with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.

    See :class:`netket.models.RNN` for explanation of the arguments related to
    the autoregressive order.
    """

    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    reorder_idx: Optional[HashableArray] = None
    """indices to transform the inputs from unordered to ordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    inv_reorder_idx: Optional[HashableArray] = None
    """indices to transform the inputs from ordered to unordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    prev_neighbors: Optional[HashableArray] = None
    """previous neighbors of each site."""
    graph: Optional[AbstractGraph] = None
    """graph of the physical system."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def reorder(self, inputs: Array, axis: int = 0) -> Array:
        return RNN.reorder(self, inputs, axis)

    def inverse_reorder(self, inputs: Array, axis: int = 0) -> Array:
        return RNN.inverse_reorder(self, inputs, axis)

    def take_prev_site(self, inputs: Array, index: int) -> Array:
        if self.reorder_idx is None:
            k = index
            prev_index = k - 1
        else:
            k = jnp.asarray(self.inv_reorder_idx)[index]
            prev_index = jnp.asarray(self.reorder_idx)[k - 1]

        inputs_i = inputs[:, prev_index, :]
        inputs_i = jnp.where(k == 0, 0, inputs_i)
        return inputs_i


class _FastLSTMNet1D(FastRNN):
    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastLSTMLayer1D(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


class _FastGRUNet1D(FastRNN):
    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastGRULayer1D(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


class _FastLSTMNet2D(FastRNN):
    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastLSTMLayer2D(
                size=self.hilbert.size,
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


@deprecate_dtype
def FastLSTMNet1D(*args, **kwargs):
    """
    1D long short-term memory network with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """
    _ensure_prev_neighbors(kwargs)
    return _FastLSTMNet1D(*args, **kwargs)


@deprecate_dtype
def FastGRUNet1D(*args, **kwargs):
    """
    1D gated recurrent unit network with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """
    _ensure_prev_neighbors(kwargs)
    return _FastGRUNet1D(*args, **kwargs)


@deprecate_dtype
def FastLSTMNet2D(*args, **kwargs):
    """
    2D long short-term memory network with snake ordering for square lattice and fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """
    _ensure_prev_neighbors(kwargs)
    return _FastLSTMNet2D(*args, **kwargs)
