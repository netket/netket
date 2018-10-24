#Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import nktools as nkt
import numpy as np
import networkx as nx

#Sigma^z*Sigma^z interactions
sigmaz = [[1, 0], [0, -1]]
mszsz = (np.kron(sigmaz, sigmaz))

#Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

#Couplings J1 and J2
J = [1, 0.4]
L = 20

pars = {}

# Define bond operators, labels, and couplings
bond_operator = [
    (J[0] * mszsz).tolist(),
    (J[1] * mszsz).tolist(),
    (-J[0] * exchange).tolist(),
    (J[1] * exchange).tolist(),
]

bond_color = [1, 2, 1, 2]

# Define custom graph
G = nx.Graph()
for i in range(L):
    G.add_edge(i, (i + 1) % L, color=1)
    G.add_edge(i, (i + 2) % L, color=2)

edge_colors = [[u, v, G[u][v]['color']] for u, v in G.edges]

# Specify custom graph
pars['Graph'] = nkt.graph(G)

#We chose a spin 1/2 hilbert space with total Sigmaz=0
pars['Hilbert'] = {
    'Name': 'Spin',
    'S': 0.5,
    'TotalSz': 0,
    'Nspins': L,
}

#defining our custom hamiltonian
pars['Hamiltonian'] = {
    'Name': 'Graph',
    'BondOps': bond_operator,
    'BondOpColors': bond_color,
}

#defining the wave function
pars['Machine'] = {
    'Name': 'RbmSpin',
    'Alpha': 1,
}

#defining the sampler
#here we use Hamiltonian sampling to preserve simmetries
pars['Sampler'] = {
    'Name': 'MetropolisHamiltonianPt',
    'Nreplicas': 16,
}

# defining the Optimizer
# here we use the Stochastic Gradient Descent
pars['Optimizer'] = {
    'Name': 'Sgd',
    'LearningRate': 0.01,
}

# defining the GroundState method
# here we use the Stochastic Reconfiguration Method
pars['GroundState'] = {
    'Method': 'Sr',
    'Nsamples': 1.0e3,
    'NiterOpt': 10000,
    'Diagshift': 0.1,
    'UseIterative': True,
    'OutputFile': "test",
}

nkt.write_input(pars, json_file="j1j2.json")
