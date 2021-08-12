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

from functools import partial

import jax
from flax.linen.initializers import *
from jax import numpy as jnp
from netket.jax.utils import dtype_real


def _compute_fans(shape, in_axis=-2, out_axis=-1):
    receptive_field_size = shape.total / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out


def _complex_uniform(key, shape, dtype):
    """
    Sample uniform random values within a disk on the complex plane,
    with zero mean and unit variance.
    """
    key_r, key_theta = jax.random.split(key)
    dtype = dtype_real(dtype)
    r = jnp.sqrt(2 * jax.random.uniform(key_r, shape, dtype))
    theta = 2 * jnp.pi * jax.random.uniform(key_theta, shape, dtype)
    return r * jnp.exp(1j * theta)


def _complex_truncated_normal(key, upper, shape, dtype):
    """
    Sample random values from a centered normal distribution on the complex plane,
    whose modulus is truncated to `upper`, and the variance before the truncation is one.
    """
    key_r, key_theta = jax.random.split(key)
    dtype = dtype_real(dtype)
    t = (1 - jnp.exp(-(upper ** 2))) * jax.random.uniform(key_r, shape, dtype)
    r = jnp.sqrt(-jnp.log(1 - t))
    theta = 2 * jnp.pi * jax.random.uniform(key_theta, shape, dtype)
    return r * jnp.exp(1j * theta)


def variance_scaling(
    scale, mode, distribution, in_axis=-2, out_axis=-1, dtype=jnp.float32
):
    """
    Initializer capable of adapting its scale to the shape of the weights tensor.

    With `distribution="truncated_normal" or "normal"`, samples are
    drawn from a truncated/untruncated normal distribution with a mean of zero and
    a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`,
    where `n` is:
    - number of input units in the weights tensor, if `mode="fan_in"`
    - number of output units, if `mode="fan_out"`
    - average of the numbers of input and output units, if `mode="fan_avg"`

    With `distribution="truncated_normal"`, the absolute values of the samples are
    truncated below 2 standard deviations before truncation.

    With `distribution="uniform"`, samples are drawn from:
    - a uniform interval, if `dtype` is real
    - a uniform disk, if `dtype` is complex
    with a mean of zero and a standard deviation of `stddev`.

    The generalization of variance scaling initializers to complex numbers is discussed in,
    e.g., `Trabelsi et. {\\it al} <https://arxiv.org/abs/1705.09792>`_.
    They proposed an axisymmetric distribution implemented by first sampling the modulus
    from the radial CDF, then uniformly sampling the phase.

    Args:
      scale: scaling factor (positive float).
      mode: one of "fan_in", "fan_out", and "fan_avg".
      distribution: random distribution to use. One of "truncated_normal",
        "normal" and "uniform".
      in_axis: axis of the input dimension in the weights tensor (default: -2).
      out_axis: axis of the output dimension in the weights tensor (default: -1).
      dtype: the dtype of the weights (default: float32).
    """

    def init(key, shape, dtype=dtype):
        shape = jax.core.as_named_shape(shape)
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode)
            )

        variance = jnp.array(scale / denominator, dtype=dtype)

        if distribution == "truncated_normal":
            if jnp.issubdtype(dtype, jnp.floating):
                # constant is stddev of standard normal truncated to (-2, 2)
                stddev = jnp.sqrt(variance) / 0.87962566103423978
                return jax.random.truncated_normal(key, -2, 2, shape, dtype) * stddev
            else:
                stddev = jnp.sqrt(variance) / 0.95311164380491208
                return _complex_truncated_normal(key, 2, shape, dtype) * stddev
        elif distribution == "normal":
            return jax.random.normal(key, shape, dtype) * jnp.sqrt(variance)
        elif distribution == "uniform":
            if jnp.issubdtype(dtype, jnp.floating):
                stddev = jnp.sqrt(3 * variance)
                return jax.random.uniform(key, shape, dtype, -1) * stddev
            else:
                return _complex_uniform(key, shape, dtype) * jnp.sqrt(variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


glorot_uniform = partial(variance_scaling, 1, "fan_avg", "uniform")
glorot_normal = partial(variance_scaling, 1, "fan_avg", "truncated_normal")
lecun_uniform = partial(variance_scaling, 1, "fan_in", "uniform")
lecun_normal = partial(variance_scaling, 1, "fan_in", "truncated_normal")
he_uniform = partial(variance_scaling, 2, "fan_in", "uniform")
he_normal = partial(variance_scaling, 2, "fan_in", "truncated_normal")

xavier_uniform = glorot_uniform
xavier_normal = glorot_normal
kaiming_uniform = he_uniform
kaiming_normal = he_normal
