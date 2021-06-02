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

import abc
from typing import Any, Callable, Iterable, Tuple, Union

import jax
from flax import linen as nn
from jax import numpy as jnp
from netket.hilbert import CustomHilbert
from netket.nn import MaskedConv1D, MaskedDense1D
from netket.nn.initializers import zeros
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import Array, DType, NNInitFunc, PyTree


class ARNN(nn.Module):
    """Base class for autoregressive neural networks."""

    @abc.abstractmethod
    def conditionals(self, inputs: Array, cache: PyTree) -> Tuple[Array, PyTree]:
        """
        Computes the probabilities for each spin to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).
          cache: auxiliary states, e.g., used to implement fast autoregressive sampling.

        Returns:
          p: the probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).
          cache: the updated cache.
        """
        raise NotImplementedError


class ARNNDense(ARNN):
    """Autoregressive neural network with dense layers."""

    hilbert: CustomHilbert
    """the discrete Hilbert space."""
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
    """the dtype of the weights (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    eps: float = 1e-7
    """a small number to avoid numerical instability."""

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            MaskedDense1D(
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

    def conditionals(self, inputs: Array, cache: PyTree) -> Tuple[Array, PyTree]:
        return _conditionals(self, inputs, cache)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


class ARNNConv1D(ARNN):
    """Autoregressive neural network with 1D convolution layers."""

    hilbert: CustomHilbert
    """the discrete Hilbert space."""
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
    """the dtype of the weights (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    eps: float = 1e-7
    """a small number to avoid numerical instability."""

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            MaskedConv1D(
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

    def conditionals(self, inputs: Array, cache: PyTree) -> Tuple[Array, PyTree]:
        return _conditionals(self, inputs, cache)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


def _conditional_log_psi(
    model: ARNN, inputs: Array, cache: PyTree
) -> Tuple[Array, PyTree]:
    """
    Computes the log of the conditional wave-functions for each spin if it takes each value.
    See `ARNN.conditionals`.
    """
    x = jnp.expand_dims(inputs, axis=-1)

    for i in range(model.layers):
        if i > 0:
            x = model.activation(x)
        x = model._layers[i](x)

    log_psi = x
    # Subtract max_log_psi_sqr when computing the norm
    log_psi_sqr = 2 * log_psi.real
    max_log_psi_sqr = log_psi_sqr.max(axis=2, keepdims=True)
    sum_psi_sqr = jnp.exp(log_psi_sqr - max_log_psi_sqr).sum(axis=2, keepdims=True)
    log_psi -= 1 / 2 * (jnp.log(sum_psi_sqr + model.eps) + max_log_psi_sqr)

    return log_psi, cache


def _conditionals(model: ARNN, inputs: Array, cache: PyTree) -> Tuple[Array, PyTree]:
    """
    Computes the probabilities for each spin to take each value. See `ARNN.conditionals`.
    """
    log_psi, cache = _conditional_log_psi(model, inputs, cache)
    p = jnp.exp(2 * log_psi.real)
    return p, cache


def _call(model: ARNN, inputs: Array) -> Array:
    """Returns log_psi."""

    if inputs.ndim == 1:
        inputs = jnp.expand_dims(inputs, axis=0)

    idx = (inputs + model.hilbert.local_size - 1) / 2
    idx = jnp.asarray(idx, jnp.int64)
    idx = jnp.expand_dims(idx, axis=-1)

    log_psi, _ = _conditional_log_psi(model, inputs, None)
    log_psi = jnp.take_along_axis(log_psi, idx, axis=-1)
    log_psi = log_psi.reshape((inputs.shape[0], -1)).sum(axis=1)

    return log_psi
