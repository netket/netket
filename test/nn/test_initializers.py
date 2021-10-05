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
import numpy as np
import pytest
from jax import numpy as jnp
from netket.jax.utils import dtype_real
from netket.nn.initializers import _complex_truncated_normal
from scipy.stats import kstest

seed = 12345


@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_complex_truncated_normal(ndim, dtype):
    np.random.seed(seed)

    key, rand_key = jax.random.split(nk.jax.PRNGKey(seed))
    # The lengths of the weight dimensions and the input dimension are random
    shape = tuple(np.random.randint(1, 10) for _ in range(ndim - 1))
    # The length of the output dimension is a statistically large number, but not too large that OOM
    len_out = int(10 ** 6 / np.prod(shape))
    shape += (len_out,)
    upper = 2
    stddev = 0.96196182800821354
    param = _complex_truncated_normal(key, upper, shape, dtype)

    assert param.shape == shape
    assert param.dtype == dtype
    assert param.mean() == pytest.approx(0, abs=2e-3)
    assert param.std() == pytest.approx(stddev, abs=1e-3)
    assert jnp.abs(param).max() == pytest.approx(upper, abs=1e-3)

    # Draw random samples using rejection sampling, and test if `param` and
    # `samples` are from the same distribution
    rand_shape = (10 ** 4,)
    rand_dtype = dtype_real(dtype)
    rand_stddev = 1 / sqrt(2)
    samples = jax.random.normal(rand_key, rand_shape, rand_dtype) * rand_stddev
    samples = samples[jnp.abs(samples) < upper]

    _, pvalue = kstest(param.flatten(), samples)
    assert pvalue > 0.01
