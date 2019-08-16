# Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.

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

# 1D Lattice
L = 20
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
# with total Sz equal to 0
hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)

# Heisenberg hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)


# Convnet machine with 7 layers (of which 3 are convolutions)
layers = (
    nk.layer.ConvolutionalHypercube(
        length=L,
        n_dim=1,
        input_channels=1,
        output_channels=2,
        stride=1,
        kernel_length=10,
        use_bias=True,
    ),
    nk.layer.Relu(input_size=2 * L),
    nk.layer.ConvolutionalHypercube(
        length=L,
        n_dim=1,
        input_channels=2,
        output_channels=2,
        stride=1,
        kernel_length=5,
        use_bias=True,
    ),
    nk.layer.Relu(input_size=(2 * L)),
    nk.layer.ConvolutionalHypercube(
        length=L,
        n_dim=1,
        input_channels=2,
        output_channels=1,
        stride=1,
        kernel_length=3,
        use_bias=True,
    ),
    nk.layer.Relu(input_size=(L)),
    nk.layer.SumOutput(input_size=(L)),
)
for layer in layers:
    layer.init_random_parameters(seed=12345, sigma=0.01)

ffnn = nk.machine.FFNN(hi, layers)

sa = nk.sampler.MetropolisExchange(machine=ffnn)


# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.01)

# Stochastic reconfiguration
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.1,
    method="Sr",
)

gs.run(output_prefix="test", n_iter=300)
