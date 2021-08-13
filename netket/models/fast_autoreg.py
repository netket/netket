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

from math import sqrt
from typing import Any, Callable, Iterable, Tuple, Union

import jax
from jax import numpy as jnp
from plum import dispatch

from netket.models.autoreg import (
    AbstractARNN,
    _call,
    _conditionals,
    _reshape_inputs,
    l2_normalize,
)
from netket.nn import FastMaskedConv1D, FastMaskedConv2D, FastMaskedDense1D
from netket.nn.initializers import zeros
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import Array, DType, NNInitFunc


class FastARNNDense(AbstractARNN):
    """
    Fast autoregressive neural network with dense layers.

    See :ref:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.

    TODO: FastMaskedDense1D does not support JIT yet, because it involves slicing the cached inputs
    and the weights with a dynamic shape.
    """

    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """number of features in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    activation: Callable[[Array], Array] = jax.nn.selu
    """the nonlinear activation function between hidden layers (default: selu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            FastMaskedDense1D(
                size=self.hilbert.size,
                features=features[i],
                exclusive=(i == 0),
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def _conditional(self, inputs: Array, index: int) -> Array:
        return _conditional(self, inputs, index)

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


class FastARNNConv1D(AbstractARNN):
    """
    Fast autoregressive neural network with 1D convolution layers.

    See :ref:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """number of features in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    kernel_size: int
    """length of the convolutional kernel."""
    kernel_dilation: int = 1
    """dilation factor of the convolution kernel (default: 1)."""
    activation: Callable[[Array], Array] = jax.nn.selu
    """the nonlinear activation function between hidden layers (default: selu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            FastMaskedConv1D(
                features=features[i],
                kernel_size=self.kernel_size,
                kernel_dilation=self.kernel_dilation,
                exclusive=(i == 0),
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def _conditional(self, inputs: Array, index: int) -> Array:
        return _conditional(self, inputs, index)

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


class FastARNNConv2D(AbstractARNN):
    """
    Fast autoregressive neural network with 2D convolution layers.

    See :ref:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """number of features in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    kernel_size: Tuple[int, int]
    """shape of the convolutional kernel `(h, w)`. Typically, `h = w // 2 + 1`."""
    kernel_dilation: Tuple[int, int] = (1, 1)
    """a sequence of 2 integers, giving the dilation factor to
    apply in each spatial dimension of the convolution kernel (default: 1)."""
    activation: Callable[[Array], Array] = jax.nn.selu
    """the nonlinear activation function between hidden layers (default: selu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""

    def setup(self):
        self.L = int(sqrt(self.hilbert.size))
        assert self.L ** 2 == self.hilbert.size

        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            FastMaskedConv2D(
                L=self.L,
                features=features[i],
                kernel_size=self.kernel_size,
                kernel_dilation=self.kernel_dilation,
                exclusive=(i == 0),
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def _conditional(self, inputs: Array, index: int) -> Array:
        return _conditional(self, inputs, index)

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


def _conditional(model: AbstractARNN, inputs: Array, index: int) -> Array:
    """
    Computes the conditional probabilities for a site to take a given value.
    See `AbstractARNN._conditional`.
    """
    if inputs.ndim == 1:
        inputs = jnp.expand_dims(inputs, axis=0)

    # When `index = 0`, it doesn't matter which site we take
    x = inputs[:, index - 1, None]

    for i in range(model.layers):
        if i > 0:
            x = model.activation(x)
        x = model._layers[i].update_site(x, index)

    log_psi = l2_normalize(x)
    p = jnp.exp(2 * log_psi.real)
    return p


@dispatch
def _reshape_inputs(model: FastARNNConv2D, inputs: Array) -> Array:  # noqa: F811
    return inputs.reshape((inputs.shape[0], model.L, model.L))
