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

import nktools as nkt

pars = {}

# defining the lattice
pars['Graph'] = {
    'Name': 'Hypercube',
    'L': 20,
    'Dimension': 1,
    'Pbc': True,
}

# defining the hamiltonian
pars['Hamiltonian'] = {
    'Name': 'Ising',
    'h': 1.0,
}

# defining the wave function
pars['Machine'] = {
    'Name': 'Jastrow',
}

# defining the sampler
# here we use Metropolis sampling with single spin flips
pars['Sampler'] = {
    'Name': 'MetropolisLocal',
}

# defining the Optimizer
# here we use the Stochastic Gradient Descent
pars['Optimizer'] = {
    'Name': 'Sgd',
    'LearningRate': 0.1,
}

# defining the GroundState method
# here we use the Stochastic Reconfiguration Method
pars['GroundState'] = {
    'Method': 'Sr',
    'Nsamples': 4000,
    'NiterOpt': 300,
    'Diagshift': 0.1,
    'UseIterative': False,
    'OutputFile': "test",
}

nkt.write_input(pars, json_file="ising1d.json")
