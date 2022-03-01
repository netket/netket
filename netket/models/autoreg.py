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
from math import sqrt
from typing import Any, Callable, Iterable, Tuple, Union

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.nn.initializers import zeros
from plum import dispatch

from netket.hilbert import Fock, Qubit, Spin
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.nn import MaskedConv1D, MaskedConv2D, MaskedDense1D
from netket.nn.masked_linear import default_kernel_init
from netket.utils.types import Array, DType, NNInitFunc


class AbstractARNN(nn.Module):
    """
    Base class for autoregressive neural networks.

    Subclasses must implement the methods `__call__` and `conditionals`.
    They can also override `_conditional` to implement the caching for fast autoregressive sampling.
    See :ref:`netket.nn.FastARNNConv1D` for example.

    They must also implement the field `machine_pow`,
    which specifies the exponent to normalize the outputs of `__call__`.
    """

    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous unconstrained Hilbert spaces are supported."""

    # machine_pow: int = 2 Must be defined on subclasses

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.hilbert, HomogeneousHilbert):
            raise ValueError(
                f"Only homogeneous Hilbert spaces are supported by ARNN, but hilbert is a {type(self.hilbert)}."
            )

        if self.hilbert.constrained:
            raise ValueError("Only unconstrained Hilbert spaces are supported by ARNN.")

    def _conditional(self, inputs: Array, index: int) -> Array:
        """
        Computes the conditional probabilities for a site to take a given value.

        It should only be called successively with indices 0, 1, 2, ...,
        as in the autoregressive sampling procedure.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).
          index: index of the site.

        Returns:
          The probabilities with dimensions (batch, Hilbert.local_size).
        """
        return self.conditionals(inputs)[:, index, :]

    @abc.abstractmethod
    def conditionals(self, inputs: Array) -> Array:
        """
        Computes the conditional probabilities for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).

        Examples:

          >>> import pytest; pytest.skip("skip automated test of this docstring")
          >>>
          >>> p = model.apply(variables, σ, method=model.conditionals)
          >>> print(p[2, 3, :])
          [0.3 0.7]
          # For the 3rd spin of the 2nd sample in the batch,
          # it takes probability 0.3 to be spin down (local state index 0),
          # and probability 0.7 to be spin up (local state index 1).
        """


class ARNNDense(AbstractARNN):
    """Autoregressive neural network with dense layers."""

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

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


class ARNNConv1D(AbstractARNN):
    """Autoregressive neural network with 1D convolution layers."""

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

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


class ARNNConv2D(AbstractARNN):
    """Autoregressive neural network with 2D convolution layers."""

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
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def setup(self):
        self.L = int(sqrt(self.hilbert.size))
        assert self.L**2 == self.hilbert.size

        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert.local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert.local_size

        self._layers = [
            MaskedConv2D(
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

    def conditionals(self, inputs: Array) -> Array:
        return _conditionals(self, inputs)

    def __call__(self, inputs: Array) -> Array:
        return _call(self, inputs)


def _normalize(log_psi: Array, machine_pow: int) -> Array:
    """
    Normalizes log_psi to have L2-norm 1 along the last axis.
    """
    return log_psi - 1 / machine_pow * jax.scipy.special.logsumexp(
        machine_pow * log_psi.real, axis=-1, keepdims=True
    )


def _conditionals_log_psi(model: AbstractARNN, inputs: Array) -> Array:
    """
    Computes the log of the conditional wave-functions for each site if it takes each value.
    See `AbstractARNN.conditionals`.
    """
    inputs = _reshape_inputs(model, inputs)

    x = jnp.expand_dims(inputs, axis=-1)

    for i in range(model.layers):
        if i > 0:
            x = model.activation(x)
        x = model._layers[i](x)

    x = x.reshape((x.shape[0], -1, x.shape[-1]))
    log_psi = _normalize(x, model.machine_pow)
    return log_psi


def _conditionals(model: AbstractARNN, inputs: Array) -> Array:
    """
    Computes the conditional probabilities for each site to take each value.
    See `AbstractARNN.conditionals`.
    """
    if inputs.ndim == 1:
        inputs = jnp.expand_dims(inputs, axis=0)

    log_psi = _conditionals_log_psi(model, inputs)

    p = jnp.exp(model.machine_pow * log_psi.real)
    return p


def _call(model: AbstractARNN, inputs: Array) -> Array:
    """Returns log_psi."""

    if inputs.ndim == 1:
        inputs = jnp.expand_dims(inputs, axis=0)

    idx = _local_states_to_numbers(model.hilbert, inputs)
    idx = jnp.expand_dims(idx, axis=-1)

    log_psi = _conditionals_log_psi(model, inputs)

    log_psi = jnp.take_along_axis(log_psi, idx, axis=-1)
    log_psi = log_psi.reshape((inputs.shape[0], -1)).sum(axis=1)
    return log_psi


@dispatch
def _reshape_inputs(model: ARNNConv2D, inputs: Array) -> Array:  # noqa: F811
    return inputs.reshape((inputs.shape[0], model.L, model.L))


@dispatch
def _reshape_inputs(model: Any, inputs: Array) -> Array:  # noqa: F811
    return inputs


@dispatch
def _local_states_to_numbers(hilbert: Spin, x: Array) -> Array:  # noqa: F811
    numbers = (x + hilbert.local_size - 1) / 2
    numbers = jnp.asarray(numbers, jnp.int32)
    return numbers


@dispatch
def _local_states_to_numbers(  # noqa: F811
    hilbert: Union[Fock, Qubit], x: Array
) -> Array:
    numbers = jnp.asarray(x, jnp.int32)
    return numbers


@dispatch
def _local_states_to_numbers(hilbert: Any, x: Array) -> Array:  # noqa: F811
    raise NotImplementedError(
        f"_local_states_to_numbers is not implemented for hilbert {type(hilbert)}."
    )
