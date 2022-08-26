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

from math import sqrt
from typing import Iterable, Optional, Union

from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.models.autoreg import AbstractARNN, _get_feature_list
from netket.models.fast_autoreg import _conditional
from netket.models.rnn import RNN, _get_snake_ordering
from netket.nn.fast_rnn import FastGRULayer1D, FastLSTMLayer1D
from netket.nn.fast_rnn_2d import FastLSTMLayer2D
from netket.nn.rnn import default_kernel_init
from netket.utils import deprecate_dtype
from netket.utils.types import Array, DType, NNInitFunc


class FastRNN(AbstractARNN):
    """
    Base class for recurrent neural networks with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    reorder_idx: Optional[jnp.ndarray] = None
    """see :class:`netket.models.AbstractARNN`."""
    inv_reorder_idx: Optional[jnp.ndarray] = None
    """see :class:`netket.models.AbstractARNN`."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def setup(self):
        raise NotImplementedError

    def _conditional(self, inputs: Array, index: int) -> Array:
        return _conditional(self, inputs, index)

    def conditionals(self, inputs: Array) -> Array:
        return RNN.conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return RNN.__call__(self, inputs)

    def reorder(self, inputs: Array) -> Array:
        return RNN.reorder(self, inputs)

    def inverse_reorder(self, inputs: Array) -> Array:
        return RNN.inverse_reorder(self, inputs)

    def _take_prev_input_site(self, inputs: Array, index: int) -> Array:
        if self.reorder_idx is None:
            k = index
            prev_index = k - 1
        else:
            k = self.inv_reorder_idx[index]
            prev_index = self.reorder_idx[k - 1]

        inputs_i = inputs[:, prev_index, :]
        zeros = jnp.zeros_like(inputs_i)
        inputs_i = jnp.where(k == 0, zeros, inputs_i)
        return inputs_i


@deprecate_dtype
class FastLSTMNet1D(FastRNN):
    """
    1D long short-term memory network with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastLSTMLayer1D(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


@deprecate_dtype
class FastGRUNet1D(FastRNN):
    """
    1D gated recurrent unit network with fast sampling.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastGRULayer1D(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


@deprecate_dtype
class FastLSTMNet2D(FastRNN):
    """2D long short-term memory network with snake ordering."""

    def setup(self):
        L = int(sqrt(self.hilbert.size))
        assert L**2 == self.hilbert.size

        if self.reorder_idx is not None or self.inv_reorder_idx is not None:
            raise ValueError("`FastLSTMNet2D` only supports snake ordering")
        reorder_idx, inv_reorder_idx = _get_snake_ordering(self.hilbert.size)

        features = _get_feature_list(self)
        self._layers = [
            FastLSTMLayer2D(
                L=L,
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=reorder_idx,
                inv_reorder_idx=inv_reorder_idx,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]
