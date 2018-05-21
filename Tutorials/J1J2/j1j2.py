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

#Sigma^z*Sigma^z interactions
sigmaz=[[1,0],[0,-1]]
mszsz=(np.kron(sigmaz,sigmaz))

#Exchange interactions
exchange=np.asarray([[0,0,0,0],[0,0,2,0],[0,2,0,0],[0,0,0,0]])

#Couplings J1 and J2
J=[1,0.4]

L=20

operators=[]
sites=[]
for i in range(L):

    for d in [0,1]:
        #\sum_i J*sigma^z(i)*sigma^z(i+d)
        operators.append((J[d]*mszsz).tolist())
        sites.append([i,(i+d+1)%L])

        #\sum_i J*(sigma^x(i)*sigma^x(i+d) + sigma^y(i)*sigma^y(i+d))
        operators.append(((-1.)**(d+1)*J[d]*exchange).tolist())
        sites.append([i,(i+d+1)%L])

pars={}

#We chose a spin 1/2 hilbert space with total Sigmaz=0
pars['Hilbert']={
    'Name'           : 'Spin',
    'S'              : 0.5,
    'TotalSz'        : 0,
    'Nspins'         : L,
}

#defining our custom hamiltonian
pars['Hamiltonian']={
    'Operators'      : operators,
    'ActingOn'       : sites,
}

#defining the wave function
pars['Machine']={
    'Name'           : 'RbmSpin',
    'Alpha'          : 1,
}

#defining the sampler
#here we use Hamiltonian sampling to preserve simmetries
pars['Sampler']={
    'Name'           : 'MetropolisHamiltonianPt',
    'Nreplicas'      : 16,
}

#defining the learning method
#here we use the Stochastic Reconfiguration Method
pars['Learning']={
    'Method'         : 'Sr',
    'Nsamples'       : 1.0e3,
    'NiterOpt'       : 10000,
    'Diagshift'      : 0.1,
    'UseIterative'   : True,
    'OutputFile'     : "test",
    'StepperType'    : 'Sgd',
    'LearningRate'   : 0.01,
}

json_file="j1j2.json"
with open(json_file, 'w') as outfile:
    json.dump(pars, outfile)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
