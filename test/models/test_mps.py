# Copyright 2024 The NetKet Authors - All rights reserved.
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
import jax.numpy as jnp


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_mps_periodic(dtype):
    L = 6
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ma = nk.models.tensor_networks.MPSPeriodic(
        hilbert=hi, bond_dim=2, param_dtype=dtype
    )
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)

    vs = nk.vqs.MCState(sa, ma)

    ha = nk.operator.IsingJax(hi, graph=g, h=1.0, dtype=dtype)
    op = nk.optimizer.Sgd(learning_rate=0.05)

    driver = nk.VMC(ha, op, variational_state=vs)

    driver.run(1)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_mps_open(dtype):
    L = 6
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ma = nk.models.tensor_networks.MPSOpen(hilbert=hi, bond_dim=2, param_dtype=dtype)
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)

    vs = nk.vqs.MCState(sa, ma)

    ha = nk.operator.IsingJax(hi, graph=g, h=1.0, dtype=dtype)
    op = nk.optimizer.Sgd(learning_rate=0.05)

    driver = nk.VMC(ha, op, variational_state=vs)

    driver.run(1)
