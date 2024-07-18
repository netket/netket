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
from typing import Any
from collections.abc import Callable

from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.models.autoreg import ARNNSequential, _get_feature_list, _normalize
from netket.nn import FastMaskedConv1D, FastMaskedConv2D, FastMaskedDense1D
from netket.nn import activation as nkactivation
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import Array, DType, NNInitFunc


class FastARNNSequential(ARNNSequential):
    """
    Implementation of a fast ARNN that sequentially calls its layers and activation function.

    Subclasses must implement `activation` as a field or a method,
    and assign a list of fast ARNN layers to `self._layers` in `setup`.

    The fast autoregressive sampling is described in `Ramachandran et. {\\it al} <https://arxiv.org/abs/1704.06001>`_.
    To generate one sample using an autoregressive network, we need to evaluate the network `N` times, where `N` is
    the number of input sites. But actually we only change one input site each time, and not all intermediate results
    depend on the changed input because of the autoregressive property, so we can cache unchanged intermediate results
    and avoid repeated computation.

    This optimization is particularly useful for convolutional neural networks (CNN) and recurrent neural networks (RNN)
    where each output site of a layer only depends on a small number of input sites, while not so useful for densely
    connected layers.
    """

    def conditional(self, inputs: Array, index: int) -> Array:
        """
        Computes the conditional probabilities for one site to take each value.
        See `AbstractARNN.conditional`.
        """
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        x = jnp.expand_dims(inputs, axis=-1)
        x = self._take_prev_site(x, index)

        for i in range(len(self._layers)):
            if i > 0 and hasattr(self, "activation"):
                x = self.activation(x)
            x = self._layers[i].update_site(x, index)

        log_psi = _normalize(x, self.machine_pow)
        p = jnp.exp(self.machine_pow * log_psi.real)
        return p

    def _take_prev_site(self, inputs: Array, index: int) -> Array:
        """
        Takes the previous site in the autoregressive order.
        """
        # When `index = 0`, it doesn't matter which site we take
        return inputs[:, index - 1]


class FastARNNDense(FastARNNSequential):
    """
    Fast autoregressive neural network with dense layers.

    See :class:`netket.models.FastARNNSequential` for a brief explanation
    of fast autoregressive sampling.

    TODO: FastMaskedDense1D does not support JIT yet, because it involves
    slicing the cached inputs and the weights with a dynamic shape.
    """

    layers: int
    """number of layers."""
    features: tuple[int, ...] | int
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    activation: Callable[[Array], Array] = nkactivation.reim_selu
    """the nonlinear activation function between hidden layers (default: reim_selu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastMaskedDense1D(
                size=self.hilbert.size,
                features=features[i],
                exclusive=(i == 0),
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


class FastARNNConv1D(FastARNNSequential):
    """
    Fast autoregressive neural network with 1D convolution layers.

    See :class:`netket.models.FastARNNSequential` for a brief
    explanation of fast autoregressive sampling.
    """

    layers: int
    """number of layers."""
    features: tuple[int, ...] | int
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    kernel_size: int
    """length of the convolutional kernel."""
    kernel_dilation: int = 1
    """dilation factor of the convolution kernel (default: 1)."""
    activation: Callable[[Array], Array] = nkactivation.reim_selu
    """the nonlinear activation function between hidden layers (default: reim_selu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            FastMaskedConv1D(
                features=features[i],
                kernel_size=self.kernel_size,
                kernel_dilation=self.kernel_dilation,
                exclusive=(i == 0),
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


class FastARNNConv2D(FastARNNSequential):
    """
    Fast autoregressive neural network with 2D convolution layers.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    layers: int
    """number of layers."""
    features: tuple[int, ...] | int
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    kernel_size: tuple[int, int]
    """shape of the convolutional kernel `(h, w)`. Typically, `h = w // 2 + 1`."""
    kernel_dilation: tuple[int, int] = (1, 1)
    """a sequence of 2 integers, giving the dilation factor to
    apply in each spatial dimension of the convolution kernel (default: 1)."""
    activation: Callable[[Array], Array] = nkactivation.reim_selu
    """the nonlinear activation function between hidden layers (default: reim_selu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def setup(self):
        self.L = int(sqrt(self.hilbert.size))
        assert self.L**2 == self.hilbert.size

        features = _get_feature_list(self)
        self._layers = [
            FastMaskedConv2D(
                L=self.L,
                features=features[i],
                kernel_size=self.kernel_size,
                kernel_dilation=self.kernel_dilation,
                exclusive=(i == 0),
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def reshape_inputs(self, inputs: Array) -> Array:
        return inputs.reshape((inputs.shape[0], self.L, self.L))
