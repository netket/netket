# Copyright 2022 The NetKet Authors - All rights reserved.
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
import netket.experimental as nkx
import optax

from .. import common

pytestmark = common.skipif_mpi


def test_find_chunk_size():
    L = 3
    graph = nk.graph.Chain(length=L)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)
    H = nk.operator.Ising(hilbert=hilbert, graph=graph, h=0)
    model = nk.models.RBM()
    sampler = nk.sampler.MetropolisLocal(hilbert)
    state = nk.vqs.MCState(sampler, model)

    optimizer = optax.sgd(learning_rate=0.1)
    driver = nk.VMC(H, variational_state=state, optimizer=optimizer)

    nk.driver.find_chunk_size(driver)

    state.chunk_size = 10**10
    nk.driver.find_chunk_size(driver)

    state.chunk_size = None
    nk.driver.find_chunk_size(driver, hilbert=hilbert)

    integrator = nkx.dynamics.Euler(dt=0.01)
    driver = nkx.TDVP(
        H,
        state,
        integrator,
        qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True),
    )

    state.chunk_size = 10**10
    nk.driver.find_chunk_size(driver)

    state.chunk_size = None
    nk.driver.find_chunk_size(driver, hilbert=hilbert)
