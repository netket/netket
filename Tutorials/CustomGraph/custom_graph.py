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
import networkx as nx


pars = {}

# defining a custom graph
# here we use networkx to generate a star graph
# and pass its edges list to NetKet
G = nx.star_graph(10)

pars['Graph'] = {
    'Edges': list(G.edges),
}

# defining the hamiltonian
pars['Hamiltonian'] = {
    'Name': 'Ising',
    'h': 1.0,
}

# defining the wave function
pars['Machine'] = {
    'Name': 'RbmSpin',
    'Alpha': 1.0,
}

# defining the sampler
# here we use Metropolis sampling with single spin flips
pars['Sampler'] = {
    'Name': 'MetropolisLocal',
}

# defining the learning method
# here we use the Stochastic Reconfiguration Method
pars['Learning'] = {
    'Method': 'Sr',
    'Nsamples': 1.0e3,
    'NiterOpt': 1000,
    'Diagshift': 0.5,
    'UseIterative': False,
    'OutputFile': "test",
    'StepperType': 'Sgd',
    'LearningRate': 0.05,
}

json_file = "custom_graph.json"
with open(json_file, 'w') as outfile:
    json.dump(pars, outfile)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
