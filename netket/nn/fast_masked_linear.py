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
from jax.nn.initializers import zeros

from netket.nn.masked_linear import (
    MaskedConv1D,
    MaskedConv2D,
    MaskedDense1D,
    default_kernel_init,
    wrap_kernel_init,
)
from netket.utils.types import Array, DType, NNInitFunc


class FastMaskedDense1D(nn.Module):
    """
    1D linear transformation module with mask for fast autoregressive NN.

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.

    TODO: FastMaskedDense1D does not support JIT yet, because it involves slicing the cached inputs
    and the weights with a dynamic shape.
    """

    size: int
    """number of sites."""
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
    def update_site(self, inputs: Array, index: int) -> Array:
        """
        Adds an input site into the cache, and applies the masked linear transformation to the cache.

        Args:
          inputs: an input site to be added into the cache with dimensions (batch, features).
          index: the index of the output site. The index of the input site should be `index - self.exclusive`.

        Returns:
          The output site with dimensions (batch, features).
        """
        dtype = jnp.promote_types(inputs.dtype, self.dtype)

        inputs = jnp.asarray(inputs, dtype)

        is_single_input = False
        if inputs.ndim == 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        batch, in_features = inputs.shape
        size = self.size

        # Number of input sites depended by the output site at the index
        size_i = index + 1

        # Initialize the cache with zeros, and the RNG key is None
        # `cache.dtype` must be the same as `inputs.dtype` (no promotion)
        _cache = self.variable(
            "cache", "inputs", zeros, None, (batch, size, in_features), inputs.dtype
        )

        initializing = self.is_mutable_collection("params")
        if not initializing:
            # Add the input site into the cache
            # To write the cache, use `_cache.value` as the left value of the assignment
            _cache.value = lax.cond(
                index - self.exclusive >= 0,
                lambda _: _cache.value.at[:, index - self.exclusive, :].set(inputs),
                lambda _: _cache.value,
                None,
            )

        cache = _cache.value
        cache = jnp.asarray(cache, dtype)

        cache_i = cache[:, :size_i, :]
        cache_i = cache_i.reshape((batch, size_i * in_features))

        # The construction of `mask` will be optimized to a constant by JIT
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

        mask_i = mask.reshape((size, in_features, size, self.features))
        mask_i = mask_i[:size_i, :, index, :]
        mask_i = mask_i.reshape((size_i * in_features, self.features))

        kernel_i = kernel.reshape((size, in_features, size, self.features))
        kernel_i = kernel_i[:size_i, :, index, :]
        kernel_i = kernel_i.reshape((size_i * in_features, self.features))

        y_i = lax.dot(cache_i, mask_i * kernel_i, precision=self.precision)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (size, self.features), self.dtype)
            bias = jnp.asarray(bias, dtype)

            bias_i = bias[index, :]

            y_i = y_i + bias_i

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
        return MaskedDense1D.__call__(self, inputs)


class FastMaskedConv1D(nn.Module):
    """
    1D convolution module with mask for fast autoregressive NN.

    The fast autoregressive sampling is described in `Ramachandran et. {\\it al} <https://arxiv.org/abs/1704.06001>`_.
    To generate one sample using an autoregressive network, we need to evaluate the network `N` times, where `N` is
    the number of input sites. But we only change one input site each time, so we can cache unchanged intermediate results
    and avoid repeated computation.
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
    dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision` for details."""
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
        dtype = jnp.promote_types(inputs.dtype, self.dtype)

        inputs = jnp.asarray(inputs, dtype)

        kernel_size = self.kernel_size - self.exclusive
        dilation = self.kernel_dilation

        is_single_input = False
        if inputs.ndim == 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        batch, in_features = inputs.shape
        assert in_features % self.feature_group_count == 0
        cache_size = kernel_size * dilation - (not self.exclusive) * (dilation - 1)

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

        initializing = self.is_mutable_collection("params")
        if not initializing:
            # Add the input site into the cache
            # To write the cache, use `_cache.value` as the left value of the assignment
            _cache.value = lax.cond(
                index - self.exclusive >= 0,
                lambda _: jnp.concatenate(
                    [_cache.value[:, 1:, :], jnp.expand_dims(inputs, axis=1)], axis=1
                ),
                lambda _: _cache.value,
                None,
            )

        cache = _cache.value
        cache = jnp.asarray(cache, dtype)

        kernel_shape = (
            kernel_size,
            in_features // self.feature_group_count,
            self.features,
        )
        kernel = self.param("kernel", self.kernel_init, kernel_shape, self.dtype)
        kernel = jnp.asarray(kernel, dtype)

        if self.exclusive and dilation > 1:
            cache = cache[:, : -(dilation - 1), :]

        dimension_numbers = flax.linen.linear._conv_dimension_numbers(cache.shape)
        y_i = lax.conv_general_dilated(
            cache,
            kernel,
            window_strides=(1,),
            padding="VALID",
            lhs_dilation=(1,),
            rhs_dilation=(dilation,),
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(bias, dtype)
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

    See :class:`netket.nn.FastMaskedConv1D` for a brief explanation of fast autoregressive sampling.
    """

    L: int
    """edge length of the 2D lattice."""
    features: int
    """number of convolution filters."""
    kernel_size: Tuple[int, int]
    """shape of the convolutional kernel `(h, w)`. Typically, :math:`h = w // 2 + 1`."""
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
    def update_site(self, inputs: Array, index: int) -> Array:
        """
        Adds an input site into the cache, and applies the masked convolution to the cache.

        Args:
          inputs: an input site to be added into the cache with dimensions (batch, features).
          index: the index of the output site. The index of the input site should be `index - self.exclusive`.

        Returns:
          The next output site with dimensions (batch, features).
        """
        dtype = jnp.promote_types(inputs.dtype, self.dtype)

        inputs = jnp.asarray(inputs, dtype)

        L = self.L
        index_w = index % L

        kernel_h, kernel_w = self.kernel_size
        dilation_h, dilation_w = self.kernel_dilation
        ones = (1, 1)

        is_single_input = False
        if inputs.ndim == 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        batch, in_features = inputs.shape
        assert in_features % self.feature_group_count == 0
        recep_h = (kernel_h - 1) * dilation_h + 1
        recep_w = (kernel_w - 1) * dilation_w + 1

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

        initializing = self.is_mutable_collection("params")
        if not initializing:
            # Add the input site into the cache
            # To write the cache, use `_cache.value` as the left value of the assignment

            inputs = jnp.expand_dims(inputs, axis=(1, 2))

            # Index of the input site in the width direction
            index_w_in = (index - self.exclusive) % L

            def _add(cache):
                # return cache.at[:, -1, index_w_in, :].set(inputs)
                return lax.dynamic_update_slice(cache, inputs, (0, -1, index_w_in, 0))

            def _shift(cache):
                return jnp.concatenate(
                    [
                        cache[:, 1:, :, :],
                        jnp.zeros((batch, 1, L, in_features), dtype=inputs.dtype),
                    ],
                    axis=1,
                )

            cache_new_row = lax.cond(
                index_w_in == 0,
                lambda _: _add(_shift(_cache.value)),
                lambda _: _shift(_add(_cache.value)),
                None,
            )

            cache_new = lax.cond(
                index_w == 0,
                lambda _: cache_new_row,
                lambda _: _add(_cache.value),
                None,
            )

            _cache.value = lax.cond(
                index - self.exclusive >= 0,
                lambda _: cache_new,
                lambda _: _cache.value,
                None,
            )

        cache = _cache.value
        cache = jnp.asarray(cache, dtype)

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
        kernel = jnp.asarray(kernel, dtype)

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

        dimension_numbers = flax.linen.linear._conv_dimension_numbers(cache.shape)
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
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(bias, dtype)
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
