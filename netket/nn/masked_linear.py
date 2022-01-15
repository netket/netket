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

from typing import Any, Tuple

import flax
from flax import linen as nn
from jax import lax
from jax import numpy as jnp
from jax.nn.initializers import lecun_normal, zeros

from netket.utils.types import Array, DType, NNInitFunc

default_kernel_init = lecun_normal()


def wrap_kernel_init(kernel_init, mask):
    """Correction to LeCun normal init."""

    corr = jnp.sqrt(mask.size / mask.sum())

    def wrapped_kernel_init(*args):
        return corr * mask * kernel_init(*args)

    return wrapped_kernel_init


class MaskedDense1D(nn.Module):
    """1D linear transformation module with mask for autoregressive NN."""

    features: int
    """number of output features, should be the last dimension."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies a masked linear transformation to the inputs.

        Args:
          inputs: input data with dimensions (batch, length, features).

        Returns:
          The transformed data.
        """
        dtype = jnp.promote_types(inputs.dtype, self.dtype)

        inputs = jnp.asarray(inputs, dtype)

        is_single_input = False
        if inputs.ndim == 2:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        batch, size, in_features = inputs.shape
        inputs = inputs.reshape((batch, size * in_features))

        mask = jnp.ones((size, size), dtype=self.dtype)
        mask = jnp.triu(mask, self.exclusive)
        mask = jnp.kron(mask, jnp.ones((in_features, self.features), dtype=self.dtype))

        kernel = self.param(
            "kernel",
            wrap_kernel_init(self.kernel_init, mask),
            (size * in_features, size * self.features),
            self.dtype,
        )
        mask = jnp.asarray(mask, dtype)
        kernel = jnp.asarray(kernel, dtype)

        y = lax.dot(inputs, mask * kernel, precision=self.precision)

        y = y.reshape((batch, size, self.features))

        if is_single_input:
            y = y.squeeze(axis=0)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (size, self.features), self.dtype)
            bias = jnp.asarray(bias, dtype)
            y = y + bias

        return y


class MaskedConv1D(nn.Module):
    """1D convolution module with mask for autoregressive NN."""

    features: int
    """number of convolution filters."""
    kernel_size: int
    """length of the convolutional kernel."""
    kernel_dilation: int
    """dilation factor of the convolution kernel."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    feature_group_count: int = 1
    """if specified, divides the input features into groups (default: 1)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the convolutional kernel."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies a masked convolution to the inputs.
        For 1D convolution, there is not really a mask. We only need to apply
        appropriate padding.

        Args:
          inputs: input data with dimensions (batch, length, features).

        Returns:
          The convolved data.
        """
        dtype = jnp.promote_types(inputs.dtype, self.dtype)

        inputs = jnp.asarray(inputs, dtype)

        kernel_size = self.kernel_size - self.exclusive
        dilation = self.kernel_dilation

        is_single_input = False
        if inputs.ndim == 2:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        in_features = inputs.shape[-1]
        assert in_features % self.feature_group_count == 0
        kernel_shape = (
            kernel_size,
            in_features // self.feature_group_count,
            self.features,
        )

        kernel = self.param("kernel", self.kernel_init, kernel_shape, self.dtype)
        kernel = jnp.asarray(kernel, dtype)

        if self.exclusive:
            inputs = inputs[:, :-dilation, :]

        # Zero padding
        y = jnp.pad(
            inputs,
            (
                (0, 0),
                ((kernel_size - (not self.exclusive)) * dilation, 0),
                (0, 0),
            ),
        )

        dimension_numbers = flax.linen.linear._conv_dimension_numbers(inputs.shape)
        y = lax.conv_general_dilated(
            y,
            kernel,
            window_strides=(1,),
            padding="VALID",
            lhs_dilation=(1,),
            rhs_dilation=(dilation,),
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if is_single_input:
            y = y.squeeze(axis=0)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(bias, dtype)
            y = y + bias

        return y


class MaskedConv2D(nn.Module):
    """2D convolution module with mask for autoregressive NN."""

    features: int
    """number of convolution filters."""
    kernel_size: Tuple[int, int]
    """shape of the convolutional kernel `(h, w)`. Typically, `h = w // 2 + 1`."""
    kernel_dilation: Tuple[int, int]
    """a sequence of 2 integers, giving the dilation factor to
    apply in each spatial dimension of the convolution kernel."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    feature_group_count: int = 1
    """if specified, divides the input features into groups (default: 1)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the convolutional kernel."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    def setup(self):
        kernel_h, kernel_w = self.kernel_size
        mask = jnp.ones((kernel_h, kernel_w, 1, 1), dtype=self.dtype)
        mask = mask.at[-1, kernel_w // 2 + (not self.exclusive) :].set(0)
        self.mask = mask

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies a masked convolution to the inputs.

        Args:
          inputs: input data with dimensions (batch, width, height, features).

        Returns:
          The convolved data.
        """
        dtype = jnp.promote_types(inputs.dtype, self.dtype)

        inputs = jnp.asarray(inputs, dtype)

        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.kernel_dilation
        ones = (1, 1)

        is_single_input = False
        if inputs.ndim == 3:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        in_features = inputs.shape[-1]
        assert in_features % self.feature_group_count == 0
        kernel_shape = self.kernel_size + (
            in_features // self.feature_group_count,
            self.features,
        )

        kernel = self.param(
            "kernel",
            wrap_kernel_init(self.kernel_init, self.mask),
            kernel_shape,
            self.dtype,
        )
        mask = jnp.asarray(self.mask, dtype)
        kernel = jnp.asarray(kernel, dtype)

        # Zero padding
        y = jnp.pad(
            inputs,
            (
                (0, 0),
                ((kernel_h - 1) * dilation_h, 0),
                (kernel_w // 2 * dilation_w, (kernel_w - 1) // 2 * dilation_w),
                (0, 0),
            ),
        )

        dimension_numbers = flax.linen.linear._conv_dimension_numbers(inputs.shape)
        y = lax.conv_general_dilated(
            y,
            mask * kernel,
            window_strides=ones,
            padding="VALID",
            lhs_dilation=ones,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if is_single_input:
            y = y.squeeze(axis=0)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(bias, dtype)
            y = y + bias

        return y
