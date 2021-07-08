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
import optax
import pytest

from .. import common

pytestmark = common.skipif_mpi


@pytest.mark.parametrize(
    "hilbert",
    [
        pytest.param(
            nk.hilbert.Spin(s=1 / 2, N=4),
            id="spin_1/2",
        ),
        pytest.param(
            nk.hilbert.Spin(s=1, N=4),
            id="spin_1",
        ),
        pytest.param(
            nk.hilbert.Fock(n_max=3, N=4),
            id="fock",
        ),
    ],
)
def test_AR_VMC(hilbert):
    graph = nk.graph.Hypercube(length=hilbert.size, n_dim=1)
    model = nk.models.ARNNDense(hilbert=hilbert, layers=3, features=5)
    sampler = nk.sampler.ARDirectSampler(hilbert, n_chains=3)

    vstate = nk.vqs.MCState(sampler, model, n_samples=6)
    assert vstate.n_discard_per_chain == 0
    vstate.sample()

    H = nk.operator.Ising(hilbert=hilbert, graph=graph, h=1)
    optimizer = optax.adam(learning_rate=1e-3)
    vmc = nk.VMC(H, optimizer, variational_state=vstate)
    vmc.run(n_iter=3)
