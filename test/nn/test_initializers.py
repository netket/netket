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

import jax
import netket as nk
import pytest
from jax import numpy as jnp
from netket.jax.utils import dtype_real
from netket.nn.initializers import lecun_normal, lecun_uniform
from numpy import prod
from scipy.stats import kstest


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
@pytest.mark.parametrize("init", ["uniform", "truncated_normal"])
def test_initializer(init, dtype):
    if init == "uniform":
        init_fun = lecun_uniform()
    elif init == "truncated_normal":
        init_fun = lecun_normal()

    key = nk.jax.PRNGKey()
    shape = (2, 2, 2, 10 ** 6)
    param = init_fun(key, shape, dtype)

    variance = 1 / prod(shape[:-1])
    stddev = sqrt(variance)

    assert param.mean() == pytest.approx(0, abs=1e-3)
    assert param.var() == pytest.approx(variance, abs=1e-2)

    if init == "uniform":
        if jnp.issubdtype(dtype, jnp.floating):
            max_norm = sqrt(3) * stddev
        else:
            max_norm = sqrt(2) * stddev
    elif init == "truncated_normal":
        if jnp.issubdtype(dtype, jnp.floating):
            max_norm = 2 / 0.87962566103423978 * stddev
        else:
            max_norm = 2 / 0.95311164380491208 * stddev

    assert jnp.abs(param).max() == pytest.approx(max_norm, abs=1e-3)

    # Draw random samples using rejection sampling, and test if `param` and
    # `samples` are from the same distribution
    rand_shape = (10 ** 3,)
    rand_dtype = dtype_real(dtype)
    if init == "uniform":
        if jnp.issubdtype(dtype, jnp.floating):
            key = nk.jax.PRNGKey()
            samples = jax.random.uniform(
                key, rand_shape, rand_dtype, -max_norm, max_norm
            )
        else:
            key_real, key_imag = jax.random.split(nk.jax.PRNGKey())
            samples = (
                jax.random.uniform(
                    key_real, rand_shape, rand_dtype, -max_norm, max_norm
                )
                + jax.random.uniform(
                    key_imag, rand_shape, rand_dtype, -max_norm, max_norm
                )
                * 1j
            )
    elif init == "truncated_normal":
        if jnp.issubdtype(dtype, jnp.floating):
            rand_stddev = max_norm / 2
        else:
            rand_stddev = max_norm / (2 * sqrt(2))
        samples = jax.random.normal(key, rand_shape, rand_dtype) * rand_stddev
    samples = samples[jnp.abs(samples) < max_norm]

    _, pvalue = kstest(param.flatten(), samples)
    assert pvalue > 0.01
