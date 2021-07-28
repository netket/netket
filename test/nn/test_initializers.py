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

import netket as nk
import pytest
from jax import numpy as jnp
from netket.nn.initializers import lecun_normal, lecun_uniform


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
@pytest.mark.parametrize("init", [lecun_uniform(), lecun_normal()])
def test_initializer(init, dtype):
    key = nk.jax.PRNGKey()
    shape = (2, 2, 2, 10 ** 6)
    param = init(key, shape, dtype)
    assert param.mean() == pytest.approx(0, abs=1e-3)
    assert param.var() == pytest.approx(1 / 8, abs=1e-2)
