# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import netket as nk
import jax.numpy as jnp

import pytest


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_Jastrow(dtype):
    N = 8
    hi = nk.hilbert.Spin(1 / 2, N)
    g = nk.graph.Chain(N)

    ma = nk.models.Jastrow(param_dtype=dtype)
    _ = ma.init(nk.jax.PRNGKey(), hi.random_state(nk.jax.PRNGKey()))

    vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), ma)

    vmc = nk.VMC(
        nk.operator.Ising(hi, g, h=1.0),
        nk.optimizer.Sgd(0.1),
        variational_state=vs,
    )
    vmc.advance(1)
