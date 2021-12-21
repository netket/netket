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

import jax
from jax import numpy as jnp
from jax.nn import initializers
from netket.jax.utils import dtype_real
from netket.utils.deprecation import deprecated


# It is left here because `netket.nn.symmetric_linear` uses it, and the one in jax is not public.
def _complex_truncated_normal(key, upper, shape, dtype):
    """
    Sample random values from a centered normal distribution on the complex plane,
    whose modulus is truncated to `upper`, and the variance before the truncation is one.
    """
    key_r, key_theta = jax.random.split(key)
    dtype = dtype_real(dtype)
    t = (1 - jnp.exp(jnp.array(-(upper ** 2), dtype))) * jax.random.uniform(
        key_r, shape, dtype
    )
    r = jnp.sqrt(-jnp.log(1 - t))
    theta = 2 * jnp.pi * jax.random.uniform(key_theta, shape, dtype)
    return r * jnp.exp(1j * theta)


_func_names = [
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "lecun_normal",
    "lecun_uniform",
    "normal",
    "ones",
    "orthogonal",
    "delta_orthogonal",
    "uniform",
    "variance_scaling",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]
_msg = "`netket.nn.initializers` is deprecated. Use `jax.nn.initializers` instead."
for func_name in _func_names:
    locals()[func_name] = deprecated(_msg, func_name)(getattr(initializers, func_name))
