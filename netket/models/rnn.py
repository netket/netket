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

from typing import Iterable, Type, Union

from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.models.autoreg import AbstractARNN, _call, _conditionals
from netket.nn.rnn import RNNLayer, default_kernel_init
from netket.utils import deprecate_dtype
from netket.utils.types import Array, DType, NNInitFunc


@deprecate_dtype
class RNN(AbstractARNN):
    """Base class for recurrent neural networks."""

    Layer: Type[RNNLayer]
    """type of all layers."""
    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            self.Layer(
                features=features[i],
                exclusive=(i == 0),
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)
