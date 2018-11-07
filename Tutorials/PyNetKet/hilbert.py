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
import pynetket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI

# #constructing a 1d lattice
g=nk.Hypercube(L=10,ndim=1)

# Hilbert space of spins from given graph
h=nk.Spin(S=0.5,graph=g)

print(h.LocalStates())
print(h.Size())

#Custom hilbert space
h=nk.CustomHilbert(local_states=[-1,0,1],graph=g)
print(h.Size())
print(h.LocalStates())

#Updating visible configurations
conf=np.array([-1.,1.,1.])
h.UpdateConf(conf,[0],[1])
print(conf)
