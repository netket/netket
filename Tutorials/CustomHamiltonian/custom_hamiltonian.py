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
import numpy as np

sigmax = [[0, 1], [1, 0]]
sigmaz = [[1, 0], [0, -1]]

mszsz = (np.kron(sigmaz, sigmaz)).tolist()

# Now we define the local operators of our hamiltonian
# And the sites on which they act
# Notice that the Transverse-Field Ising model as defined here has sign problem
operators = []
sites = []
L = 20
for i in range(L):
    # \sum_i sigma^x(i)
    operators.append(sigmax)
    sites.append([i])
    # \sum_i sigma^z(i)*sigma^z(i+1)
    operators.append(mszsz)
    sites.append([i, (i + 1) % L])

pars = {}

# first we choose a hilbert space for our custom hamiltonian
pars['Hilbert'] = {
    'QuantumNumbers': [1, -1],
    'Size': L,
}

# defining a custom hamiltonian
pars['Hamiltonian'] = {
    'Operators': operators,
    'ActingOn': sites,
}

# defining the wave function
pars['Machine'] = {
    'Name': 'RbmSpin',
    'Alpha': 1,
}

# defining the sampler
# here we use Metropolis sampling with single spin flips
pars['Sampler'] = {
    'Name': 'MetropolisLocal',
}

# defining the learning method
# here we use a Gradient Descent with AdaMax
pars['Learning'] = {
    'Method': 'Gd',
    'Nsamples': 1.0e3,
    'NiterOpt': 40000,
    'OutputFile': "test",
    'StepperType': 'AdaMax',
}

json_file = "custom_hamiltonian.json"
with open(json_file, 'w') as outfile:
    json.dump(pars, outfile)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
