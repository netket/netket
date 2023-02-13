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
from typing import Any, Callable, Tuple, Union

import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.nn import MaskedConv1D, MaskedConv2D, MaskedDense1D
from netket.nn.masked_linear import default_kernel_init
from netket.nn import activation as nkactivation
from netket.utils.types import Array, DType, NNInitFunc
from netket.utils import deprecate_dtype


class AbstractARNN(nn.Module):
    """
    Base class for autoregressive neural networks.

    Subclasses must implement the method `conditionals_log_psi`, or override the methods
    `__call__` and `conditional` if desired.

    They can override `conditional` to implement the caching for fast autoregressive sampling.
    See :class:`netket.nn.FastARNNConv1D` for example.

    They must also implement the field `machine_pow`,
    which specifies the exponent to normalize the outputs of `__call__`.
    """

    hilbert: HomogeneousHilbert
    """the Hilbert space. Only homogeneous unconstrained Hilbert spaces are supported."""

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.hilbert, HomogeneousHilbert):
            raise ValueError(
                f"Only homogeneous Hilbert spaces are supported by ARNN, but hilbert is a {type(self.hilbert)}."
            )

        if self.hilbert.constrained:
            raise ValueError("Only unconstrained Hilbert spaces are supported by ARNN.")

    @abc.abstractmethod
    def conditionals_log_psi(self, inputs: Array) -> Array:
        """
        Computes the log of the conditional wave-functions for each site to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The log psi with dimensions (batch, Hilbert.size, Hilbert.local_size).
        """

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
        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        log_psi = self.conditionals_log_psi(inputs)

        p = jnp.exp(self.machine_pow * log_psi.real)
        return p

    def conditional(self, inputs: Array, index: int) -> Array:
        """
        Computes the conditional probabilities for one site to take each value.

        It should only be called successively with indices 0, 1, 2, ...,
        as in the autoregressive sampling procedure.

        Args:
          inputs: configurations of partially sampled sites with dimensions (batch, Hilbert.size),
            where the sites that `index` depends on must be already sampled.
          index: index of the site being queried.

        Returns:
          The probabilities with dimensions (batch, Hilbert.local_size).
        """
        # TODO: remove this in future
        if hasattr(self, "_conditional"):
            from netket.utils import warn_deprecation

            warn_deprecation(
                "AbstractARNN._conditional has been renamed to AbstractARNN.conditional "
                "as a public API. Please update your subclass to use fast AR sampling."
            )
            return self._conditional(inputs, index)

        return self.conditionals(inputs)[:, index, :]

    def __call__(self, inputs: Array) -> Array:
        """
        Computes the log wave-functions for input configurations.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          The log psi with dimension (batch,).
        """

        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        idx = self.hilbert.states_to_local_indices(inputs)
        idx = jnp.expand_dims(idx, axis=-1)

        log_psi = self.conditionals_log_psi(inputs)

        log_psi = jnp.take_along_axis(log_psi, idx, axis=-1)
        log_psi = log_psi.reshape((inputs.shape[0], -1)).sum(axis=1)
        return log_psi


class ARNNSequential(AbstractARNN):
    """
    Implementation of an ARNN that sequentially calls its layers and activation function.

    Subclasses must implement `activation` as a field or a method,
    and assign a list of ARNN layers to `self._layers` in `setup`.

    Note:
        If you want to use real parameters and output a complex wave function, such as in
        `Hibat-Allah et. {\\it al} <https://arxiv.org/abs/2002.02973>`_,
        you can implement `conditionals_log_psi` differently, compute the modulus and the phase
        using the output of the last RNN layer, and combine them into the wave function.

        During the sampling, `conditionals_log_psi` is called and only the modulus is
        needed, so the computation of the phase becomes an overhead. To avoid this
        overhead, you can override `conditional` and only compute the modulus there.
    """

    def conditionals_log_psi(self, inputs: Array) -> Array:
        inputs = self.reshape_inputs(inputs)

        x = jnp.expand_dims(inputs, axis=-1)

        for i in range(len(self._layers)):
            if i > 0:
                x = self.activation(x)
            x = self._layers[i](x)

        x = x.reshape((x.shape[0], -1, x.shape[-1]))
        log_psi = _normalize(x, self.machine_pow)
        return log_psi

    def reshape_inputs(model: Any, inputs: Array) -> Array:
        """
        Reshapes the inputs from (batch_size, hilbert_size) to (batch_size, spatial_dims...)
        before sending them to the ARNN layers.
        """
        return inputs


@deprecate_dtype
class ARNNDense(ARNNSequential):
    """Autoregressive neural network with dense layers."""

    layers: int
    """number of layers."""
    features: Union[Tuple[int, ...], int]
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
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


@deprecate_dtype
class ARNNConv1D(ARNNSequential):
    """Autoregressive neural network with 1D convolution layers."""

    layers: int
    """number of layers."""
    features: Union[Tuple[int, ...], int]
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
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


class ARNNConv2D(ARNNSequential):
    """Autoregressive neural network with 2D convolution layers."""

    layers: int
    """number of layers."""
    features: Union[Tuple[int, ...], int]
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    kernel_size: Tuple[int, int]
    """shape of the convolutional kernel `(h, w)`. Typically, `h = w // 2 + 1`."""
    kernel_dilation: Tuple[int, int] = (1, 1)
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
                param_dtype=self.param_dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def reshape_inputs(self, inputs: Array) -> Array:
        return inputs.reshape((inputs.shape[0], self.L, self.L))


def _normalize(log_psi: Array, machine_pow: int) -> Array:
    """
    Normalizes log_psi to have L2-norm 1 along the last axis.
    """
    return log_psi - 1 / machine_pow * jax.scipy.special.logsumexp(
        machine_pow * log_psi.real, axis=-1, keepdims=True
    )
