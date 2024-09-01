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

from typing import Any

from flax import linen as nn
from flax.linen.dtypes import promote_dtype

from jax import lax
from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.nn.masked_linear import (
    MaskedConv1D,
    MaskedConv2D,
    MaskedDense1D,
    _conv_dimension_numbers,
    default_kernel_init,
    wrap_kernel_init,
)
from netket.utils.types import Array, DType, NNInitFunc


class FastMaskedDense1D(nn.Module):
    """
    1D linear transformation module with mask for fast autoregressive NN.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.

    TODO: FastMaskedDense1D does not support JIT yet, because it involves slicing the cached inputs
    and the weights with a dynamic shape.
    """

    size: int
    """number of sites."""
    features: int
    """output feature density, should be the last dimension."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    @nn.compact
    def update_site(self, inputs: Array, index: int) -> Array:
        """
        Adds an input site into the cache, and applies the masked linear transformation to the cache.

        Args:
          inputs: an input site to be added into the cache with dimensions (batch, features).
          index: the index of the output site. The index of the input site should be `index - self.exclusive`.

        Returns:
          The output site with dimensions (batch, features).
        """
        if inputs.ndim == 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)
        else:
            is_single_input = False

        batch, in_features = inputs.shape
        size = self.size

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (size, self.features), self.param_dtype
            )
        else:
            bias = None

        # The construction of `mask` will be optimized to a constant by JIT
        mask = jnp.ones((size, size), dtype=self.param_dtype)
        mask = jnp.triu(mask, self.exclusive)
        mask = jnp.kron(
            mask, jnp.ones((in_features, self.features), dtype=self.param_dtype)
        )

        kernel = self.param(
            "kernel",
            wrap_kernel_init(self.kernel_init, mask),
            (size * in_features, size * self.features),
            self.param_dtype,
        )

        inputs, kernel, mask, bias = promote_dtype(
            inputs, kernel, mask, bias, dtype=None
        )

        # Number of input sites depended by the output site at the index
        size_i = index + 1

        # Initialize the cache with zeros, and the RNG key is None
        # `cache.dtype` must be the same as `inputs.dtype` (no promotion)
        _cache = self.variable(
            "cache", "inputs", zeros, None, (batch, size, in_features), inputs.dtype
        )

        if not self.is_initializing():
            # Add the input site into the cache
            # To write the cache, use `_cache.value` as the left value of the assignment
            _cache.value = jnp.where(
                index - self.exclusive >= 0,
                _cache.value.at[:, index - self.exclusive, :].set(inputs),
                _cache.value,
            )

        cache = _cache.value

        cache_i = cache[:, :size_i, :]
        cache_i = cache_i.reshape((batch, size_i * in_features))

        mask_i = mask.reshape((size, in_features, size, self.features))
        mask_i = mask_i[:size_i, :, index, :]
        mask_i = mask_i.reshape((size_i * in_features, self.features))

        kernel_i = kernel.reshape((size, in_features, size, self.features))
        kernel_i = kernel_i[:size_i, :, index, :]
        kernel_i = kernel_i.reshape((size_i * in_features, self.features))

        y_i = lax.dot(cache_i, mask_i * kernel_i, precision=self.precision)

        if self.use_bias:
            y_i = y_i + bias[index, :]  # type: ignore[assignment,index]

        assert y_i.shape[1] == self.features

        if is_single_input:
            y_i = y_i.squeeze(axis=0)

        return y_i

    def __call__(self, inputs: Array) -> Array:
        """
        Applies the masked linear transformation to all input sites.

        Args:
          inputs: input data with dimensions (batch, size, features).

        Returns:
          The transformed data.
        """
        return MaskedDense1D.__call__(self, inputs)  # type: ignore[arg-type]


class FastMaskedConv1D(nn.Module):
    """
    1D convolution module with mask for fast autoregressive NN.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

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
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the convolutional kernel."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    @nn.compact
    def update_site(self, inputs: Array, index: int) -> Array:
        """
        Adds an input site into the cache, and applies the masked convolution to the cache.

        Args:
          inputs: an input site to be added into the cache with dimensions (batch, features).
          index: the index of the output site. The index of the input site should be `index - self.exclusive`.

        Returns:
          The next output site with dimensions (batch, features).
        """
        kernel_size = self.kernel_size - self.exclusive
        dilation = self.kernel_dilation

        if inputs.ndim == 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)
        else:
            is_single_input = False

        batch, in_features = inputs.shape
        assert in_features % self.feature_group_count == 0
        cache_size = kernel_size * dilation - (not self.exclusive) * (dilation - 1)

        kernel_shape = (
            kernel_size,
            in_features // self.feature_group_count,
            self.features,
        )
        kernel = self.param("kernel", self.kernel_init, kernel_shape, self.param_dtype)

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=None)

        # Initialize the cache with zeros, and the RNG key is None
        # `cache.dtype` must be the same as `inputs.dtype` (no promotion)
        _cache = self.variable(
            "cache",
            "inputs",
            zeros,
            None,
            (batch, cache_size, in_features),
            inputs.dtype,
        )

        if not self.is_initializing():
            # Add the input site into the cache
            # To write the cache, use `_cache.value` as the left value of the assignment
            _cache.value = jnp.where(
                index - self.exclusive >= 0,
                jnp.concatenate(
                    [_cache.value[:, 1:, :], jnp.expand_dims(inputs, axis=1)], axis=1
                ),
                _cache.value,
            )

        cache = _cache.value

        if self.exclusive and dilation > 1:
            cache = cache[:, : -(dilation - 1), :]

        dimension_numbers = _conv_dimension_numbers(cache.shape)
        y_i = lax.conv_general_dilated(
            cache,
            kernel,  # type: ignore[arg-type]
            window_strides=(1,),
            padding="VALID",
            lhs_dilation=(1,),
            rhs_dilation=(dilation,),
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if self.use_bias:
            y_i = y_i + bias

        y_i = y_i.squeeze(axis=1)

        if is_single_input:
            y_i = y_i.squeeze(axis=0)

        return y_i

    def __call__(self, inputs: Array) -> Array:
        """
        Applies the masked convolution to all input sites.

        Args:
          inputs: input data with dimensions (batch, size, features).

        Returns:
          The convolved data.
        """
        return MaskedConv1D.__call__(self, inputs)


class FastMaskedConv2D(nn.Module):
    """
    2D convolution module with mask for fast autoregressive NN.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    L: int
    """edge length of the 2D lattice."""
    features: int
    """number of convolution filters."""
    kernel_size: tuple[int, int]
    """shape of the convolutional kernel `(h, w)`. Typically, :math:`h = w // 2 + 1`."""
    kernel_dilation: tuple[int, int]
    """a sequence of 2 integers, giving the dilation factor to
    apply in each spatial dimension of the convolution kernel."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    feature_group_count: int = 1
    """if specified, divides the input features into groups (default: 1)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the convolutional kernel."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    def setup(self):
        MaskedConv2D.setup(self)

    @nn.compact
    def update_site(self, inputs: Array, index: int) -> Array:
        """
        Adds an input site into the cache, and applies the masked convolution to the cache.

        Args:
          inputs: an input site to be added into the cache with dimensions (batch, features).
          index: the index of the output site. The index of the input site should be `index - self.exclusive`.

        Returns:
          The next output site with dimensions (batch, features).
        """
        L = self.L
        index_w = index % L

        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.kernel_dilation
        ones = (1, 1)

        if inputs.ndim == 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)
        else:
            is_single_input = False

        batch, in_features = inputs.shape
        assert in_features % self.feature_group_count == 0
        recep_h = (kernel_h - 1) * dilation_h + 1
        recep_w = (kernel_w - 1) * dilation_w + 1

        if self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        kernel_shape = self.kernel_size + (
            in_features // self.feature_group_count,
            self.features,
        )
        kernel = self.param(
            "kernel",
            wrap_kernel_init(self.kernel_init, self.mask),
            kernel_shape,
            self.param_dtype,
        )

        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=None)

        # Initialize the cache with zeros, and the RNG key is None
        # `cache.dtype` must be the same as `inputs.dtype` (no promotion)
        _cache = self.variable(
            "cache",
            "inputs",
            zeros,
            None,
            (batch, recep_h, L, in_features),
            inputs.dtype,
        )

        if not self.is_initializing():
            # Add the input site into the cache
            # To write the cache, use `_cache.value` as the left value of the assignment

            inputs = jnp.expand_dims(inputs, axis=(1, 2))

            # Index of the input site in the width direction
            index_w_in = (index - self.exclusive) % L

            def _add(cache):
                # return cache.at[:, -1, index_w_in, :].set(inputs)
                return lax.dynamic_update_slice(cache, inputs, (0, -1, index_w_in, 0))

            def _shift(cache):
                return jnp.pad(cache[:, 1:, :, :], ((0, 0), (0, 1), (0, 0), (0, 0)))

            cache_new_row = jnp.where(
                index_w_in == 0, _add(_shift(_cache.value)), _shift(_add(_cache.value))
            )

            cache_new = jnp.where(index_w == 0, cache_new_row, _add(_cache.value))

            _cache.value = jnp.where(
                index - self.exclusive >= 0, cache_new, _cache.value
            )

        cache = _cache.value

        # Zero padding
        cache = jnp.pad(
            cache,
            (
                (0, 0),
                (0, 0),
                (kernel_w // 2 * dilation_w, (kernel_w - 1) // 2 * dilation_w),
                (0, 0),
            ),
        )

        # cache = cache[:, :, index_w : index_w + recep_w, :]
        cache = lax.dynamic_slice(
            cache, (0, 0, index_w, 0), (batch, recep_h, recep_w, in_features)
        )

        dimension_numbers = _conv_dimension_numbers(cache.shape)
        y_i = lax.conv_general_dilated(
            cache,
            kernel,
            window_strides=ones,
            padding="VALID",
            lhs_dilation=ones,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if self.use_bias:
            y_i = y_i + bias

        y_i = y_i.squeeze(axis=(1, 2))

        if is_single_input:
            y_i = y_i.squeeze(axis=0)

        return y_i

    def __call__(self, inputs: Array) -> Array:
        """
        Applies the masked convolution to all input sites.

        Args:
          inputs: input data with dimensions (batch, width, height, features).

        Returns:
          The convolved data.
        """
        return MaskedConv2D.__call__(self, inputs)
