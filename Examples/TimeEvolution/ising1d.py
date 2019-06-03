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
import json

pars = {}

# defining the hilbert space
pars["Hilbert"] = {"Name": "Spin", "S": 0.5}

# defining the lattice
pars["Graph"] = {"Name": "Hypercube", "L": 10, "Dimension": 1, "Pbc": True}

# defining the hamiltonian
pars["Hamiltonian"] = {"Name": "Ising", "h": 1.0}

# define two initial states
initial_states = []
fraction = 1 / 1024
# state 0: uniform amplitude over all basis states
initial_states.append([[fraction, 0.0] for _ in range(1024)])
# state 1: exactly the first basis state
initial_states.append([[1.0, 0.0]] + [[0.0, 0.0] for _ in range(1023)])


# Specify the parameters for the time evolution
pars["TimeEvolution"] = {
    "TimeStepper": "Dopri54",
    "AbsTol": 1e-9,
    "RelTol": 1e-9,
    "MatrixWrapper": "Sparse",
    "StartTime": 0.0,
    "EndTime": 10.0,
    "TimeStep": 0.5,
    "OutputFiles": "ising1d_output_%i.txt",
    # Specifiy a set of initial configurations to propagate
    "InitialStates": initial_states,
}

json_file = "ising1d.json"
with open(json_file, "w") as outfile:
    json.dump(pars, outfile, indent=4)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
