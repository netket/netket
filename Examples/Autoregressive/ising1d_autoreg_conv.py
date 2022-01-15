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

# 1D Lattice
L = 20  # 10

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin Hamiltonian
ha = nk.operator.Ising(hilbert=hi, graph=g, h=1)

# Autoregressive neural network
ma = nk.models.ARNNConv1D(hilbert=hi, layers=2, features=10, kernel_size=10)

# Autoregressive direct sampling
sa = nk.sampler.ARDirectSampler(hi)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.1)

# SR
sr = nk.optimizer.SR(diag_shift=0.01)

# Variational state
# With direct sampling, we don't need many samples in each step to form a
# Markov chain, and we don't need to discard samples
vs = nk.vqs.MCState(sa, ma, n_samples=64)

# n_parameters also takes masked parameters into account
# The number of non-masked parameters is about a half
print("n_parameters:", vs.n_parameters)

# Variational monte carlo driver with a variational state
gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

# Run the optimization for 1000 iterations
gs.run(n_iter=1000, out="test")

# Use more samples to estimate the Hamiltonian and the error bar
vs.n_samples = 1000
print(vs.expect(ha))
