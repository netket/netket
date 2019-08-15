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

# Tutorial for defining a custom sampler
# taking L=20 Heisenberg spin chain as example
# (adapted from Heisenberg1d tutorial)
# (you can use plot_heis.py from Heisenberg1d folder identically here to plot results)

L = 20

# defining the hilbert space
pars["Hilbert"] = {"Name": "Spin", "S": 0.5}

# defining the lattice
pars["Graph"] = {"Name": "Hypercube", "L": L, "Dimension": 1, "Pbc": True}

# defining the hamiltonian
pars["Hamiltonian"] = {"Name": "Heisenberg"}

# defining the wave function
pars["Machine"] = {"Name": "RbmSpinSymm", "Alpha": 1.0}

# defining the custom sampler
# here we use two types of moves : local spin flip, and exchange flip between two sites
# note that each line and column have to add up to 1.0 (stochastic matrices)
# we also choose a relative frequency of 2 for local-spin flips with respect to exchange flips
spin_flip = [[0, 1], [1, 0]]
exchange_flip = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
weight_spin_flip = 1.0
weight_exchange_flip = 2.0

# adding both types of flip for all sites in the chain
operators = []
sites = []
weights = []
for i in range(L):
    operators.append(exchange_flip)
    sites.append([i, (i + 1) % L])
    weights.append(weight_exchange_flip)
    operators.append(spin_flip)
    sites.append([i])
    weights.append(weight_spin_flip)

# now we define the custom sampler accordingly
pars["Sampler"] = {
    "MoveOperators": operators,
    "ActingOn": sites,
    "MoveWeights": weights,
    # parallel tempering is also possible with custom sampler (uncomment the following line)
    #'Nreplicas' : 12,
}

# defining the Optimizer
# here we use AdaMax
pars["Optimizer"] = {"Name": "AdaMax"}

# defining the GroundState method
# here we use the Stochastic Reconfiguration Method
pars["GroundState"] = {
    "Method": "Sr",
    "Nsamples": 1.0e3,
    "NiterOpt": 4000,
    "Diagshift": 0.1,
    "UseIterative": False,
    "OutputFile": "test",
}

json_file = "customsampler_heisenberg1d.json"
with open(json_file, "w") as outfile:
    json.dump(pars, outfile)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
