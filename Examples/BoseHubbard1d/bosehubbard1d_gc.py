# Copyright 2018-2020 The Simons Foundation, Inc. - All Rights Reserved.

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
from mpi4py import MPI
from scipy.sparse.linalg import eigsh

# 1D Periodic Lattice
g = nk.graph.Hypercube(length=5, n_dim=1, pbc=True)

# Boson Hilbert Space
hi = nk.hilbert.Boson(graph=g, n_max=4)

# Bose Hubbard Hamiltonian
ha = nk.operator.BoseHubbard(U=3.2, hilbert=hi, mu=0.0)

# Use scipy sparse diagonalization
vals, vecs = eigsh(ha.to_sparse(), k=1, which="SA")
print("eigenvalues with scipy sparse:", vals)

# RbmMultiVal with Symmetry
ma = nk.machine.RbmMultiVal(alpha=1, hilbert=hi, symmetry=True)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Exact Sampler for testing
sa = nk.sampler.ExactSampler(machine=ma)

# Stochastic gradient descent optimization
op = nk.optimizer.Sgd(ma, 0.01)

# Variational Monte Carlo
vmc = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=1000,
    diag_shift=0.1,
    use_iterative=True,
    method="Sr",
)

vmc.run(output_prefix="test", n_iter=4000)
