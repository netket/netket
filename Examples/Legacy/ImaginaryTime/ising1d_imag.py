# Copyright 2020 The Netket Authors. - All Rights Reserved.
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

if nk.MPI.size() > 1:
    import sys

    if nk.MPI.rank() == 0:
        print(
            "Error: The exact imaginary time propagation currently "
            "only supports one MPI process"
        )
    sys.exit(1)


L = 14

# defining the lattice
graph = nk.graph.Hypercube(L, n_dim=1, pbc=True)

# defining the hilbert space
hilbert = nk.hilbert.Spin(graph, 0.5)
n_states = hilbert.n_states

# defining the hamiltonian and wrap it as matrix
hamiltonian = nk.operator.Ising(hilbert, h=1.0)

# add observable
obs = {"Sz[0]": nk.operator.spin.sigmaz(hilbert, 0)}

# run from random initial state (does not need to be normalized, this is done
# by the driver)
import numpy as np

psi0 = 0.01 * np.random.rand(n_states) + 0.01j * np.random.rand(n_states)

# create ground state driver
driver = nk.exact.PyExactTimePropagation(
    hamiltonian, t0=0.0, dt=0.05, initial_state=psi0, propagation_type="imaginary"
)

# Run time time propagation
driver.run("ising_imag", 500, obs=obs, write_every=1, save_params_every=1)
