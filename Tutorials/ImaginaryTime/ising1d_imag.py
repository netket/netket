# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

from mpi4py import MPI
import netket as nk


L = 20

# defining the lattice
graph = nk.graph.Hypercube(L, ndim=1, pbc=True)

# defining the hilbert space
hilbert = nk.hilbert.Spin(graph, 0.5)

# defining the hamiltonian and wrap it as matrix
hamiltonian = nk.operator.Ising(hilbert, h=1.0)
mat = nk.operator.wrap_as_matrix(hamiltonian)

# create time stepper
stepper = nk.dynamics.create_timestepper(mat.dimension, rel_tol=1e-10, abs_tol=1e-10)

# prepare output
output = nk.output.JsonOutputWriter('test.log', 'test.wf')

# create ground state driver
driver = nk.ImaginaryTimeDriver(mat, stepper, output, tmin=0, tmax=20, dt=0.1)

# add observable (TODO: more interesting observable)
driver.add_observable(hamiltonian, 'Hamiltonian')

# run from random initial state (does not need to be normalized, this is done 
# by the driver)

import numpy as np
init = np.random.rand(mat.dimension)

driver.run(init)

