# Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.
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

from netket import legacy as nk

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
# with total Sz equal to 0
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes, total_sz=0)

# Heisenberg hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)


# Fully-connected machine
layers = (
    nk.layer.FullyConnected(input_size=L, output_size=(2 * L), use_bias=True),
    nk.layer.Lncosh(input_size=(2 * L)),
    nk.layer.SumOutput(input_size=(2 * L)),
)
for layer in layers:
    layer.init_random_parameters(seed=12345, sigma=0.01)

ffnn = nk.machine.FFNN(hi, layers)

sa = nk.sampler.MetropolisExchange(machine=ffnn)


# Optimizer
op = nk.optimizer.Sgd(ma, learning_rate=0.05)

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.1,
    method="Sr",
    use_iterative=True,
)

gs.run(out="test", n_iter=300)
