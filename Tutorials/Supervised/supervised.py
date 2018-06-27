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


from __future__ import print_function
import json
import numpy as np

# Couplings J1 and J2
# J = [1, 0.4]

L = 6

pars = {}

# We can either provide the Hilbert information in the InputFile
# or to provide the information here.
# Both way should be fine.
# But it would affect the interface in Supervised, Data class

#We chose a spin 1/2 hilbert space with total Sigmaz=0
pars['Hilbert'] = {
    'Name': 'Spin',
    'S': 0.5,
    'TotalSz': 0,
    'Nspins': L,
}

#defining the wave function
pars['Machine'] = {
    'Name': 'RbmSpin',
    'Alpha': 1,
}

#defining the Supervised learning method
pars['Supervised'] = {
    'Loss' : 'L2',
    'Nsamples': 1.0e3,
    'NiterOpt': 10000,
    'InputFile': "psi",
    'OutputFile': "test",
    'StepperType': 'Sgd',
    'LearningRate': 0.01,
}

json_file = "supervised.json"
with open(json_file, 'w') as outfile:
    json.dump(pars, outfile)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
