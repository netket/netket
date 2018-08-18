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

sigmaz = [[1, 0, 0], [0,0,0], [0, 0, -1]]
sigmaplus = [[0,0,0],[np.sqrt(2.),0,0],[0,np.sqrt(2),0]]
sigmaminus = [[0,np.sqrt(2),0],[0,0,np.sqrt(2.)],[0,0,0]]

heisenberg=np.kron(sigmaz,sigmaz)+ \
           0.5*np.kron(sigmaplus,sigmaminus)+\
           0.5*np.kron(sigmaminus,sigmaplus)

heisenberg_squared = (np.linalg.matrix_power(heisenberg,2))

# System size
L = 20

pars = {}
pars['Graph'] = {
    'Name': 'Hypercube',
    'L': L,
    'Dimension': 1,
    'Pbc': True,
}

# Setting up the Hilbert space
pars['Hilbert'] = {
    'S': 1,
    'Nspins': L,
    'TotalSz':0,
    'Name': "Spin",
}

# defining a custom bond hamiltonian
pars['Hamiltonian'] = {
    'Name': 'Graph',
    'BondOps': [(heisenberg).tolist(),(heisenberg_squared/3.).tolist()],
}

# defining the wave function
pars['Machine'] = {
    'Name': 'FFNN',
    'Layers': [{'Name':'Recurrent', 'HiddenUnits':4, 'Inputs': 20, 'Outputs':20, "UseBias":True, 'Activation':'Tanh' }],
}

# defining the sampler
# here we use Metropolis sampling with single spin flips
pars['Sampler'] = {
    'Name': 'MetropolisLocal',
}

# defining the Optimizer
# here we use AdaMax
pars['Optimizer'] = {
    'Name': 'AdaMax',
}

# defining the GroundState method
# here we use a Gradient Descent with AdaMax
pars['GroundState'] = {
    'Method': 'Gd',
    'Nsamples': 1.0e3,
    'NiterOpt': 40000,
    'OutputFile': "test",
}

json_file = "AKLT.json"
with open(json_file, 'w') as outfile:
    json.dump(pars, outfile)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
