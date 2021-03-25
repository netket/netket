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
import pytest


# This might need to be refactored into a separate file since the files would have a lot of common code from the below
# class
class Example:
    def __init__(self, L, n_samp, alpha, dimensions):
        self.lattice = nk.graph.Hypercube(length=L, n_dim=dimensions, pbc=True)
        self.hilbertSpace = nk.hilbert.Spin(s=1 / 2, N=self.lattice.n_nodes)
        # self.operator = nk.operator.Ising(hilbert=self.hilbertSpace, graph=self.lattice, h=1.0)  # Is this even needed?
        self.rbmachine = nk.models.RBM(alpha=alpha, dtype=complex)
        self.sampler = nk.sampler.MetropolisLocal(self.hilbertSpace, n_chains=32)
        self.state = nk.variational.MCState(
            self.sampler, self.rbmachine, n_samples=n_samp, seed=123
        )


@pytest.fixture(scope="module", params=[[64, 1024, 1, 1]])
def example(request):
    return Example(
        L=request.param[0],
        n_samp=request.param[1],
        alpha=request.param[2],
        dimensions=request.param[3],
    )


@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.benchmark(min_rounds=10)
def test_sr_solver(example, benchmark, centered):
    @benchmark
    def sr_solver():
        qgt = example.state.quantum_geometric_tensor(
            nk.optimizer.SR(diag_shift=0.01, centered=centered)
        )
        return (qgt @ example.state.parameters)["Dense"]["kernel"].block_until_ready()

    assert True
