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
import numpy as np
import networkx as nx

sigmax = [[0, 1], [1, 0]]
sigmaz = [[1, 0], [0, -1]]

mszsz = (np.kron(sigmaz, sigmaz)).tolist()

# Now we define the local operators of our hamiltonian
# And the sites on which they act
# Notice that the Transverse-Field Ising model as defined here has sign problem
L = 20
site_operator = [sigmax]
bond_operator = [mszsz]

# Defining a custom graph
G = nx.Graph()
for i in range(L):
    G.add_edge(i, (i + 1) % L)

pars = {}

pars['Graph'] = nkt.graph(G)

# first we choose a hilbert space for our custom hamiltonian
pars['Hilbert'] = {
    'QuantumNumbers': [1, -1],
    'Size': len(list(G.edges)),
}

# defining a graph hamiltonian
pars['Hamiltonian'] = {
    'Name': 'Graph',
    'SiteOps': site_operator,
    'BondOps': bond_operator,
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

nkt.write_input(pars, json_file="ising.json")
