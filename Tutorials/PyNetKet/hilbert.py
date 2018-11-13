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
import netket as nk
from netket.hilbert import *
import networkx as nx
import numpy as np
from mpi4py import MPI

# #constructing a 1d lattice
g = nk.graph.Hypercube(L=10, ndim=1)

# Hilbert space of spins from given graph
h = Spin(s=0.5, graph=g)

print(h.local_states())
print(h.size())

#Custom hilbert space
h = CustomHilbert(local_states=[-1, 0, 1], graph=g)
print(h.size())
print(h.local_states())

#Updating visible configurations
conf = np.array([-1., 1., 1.])
h.update_conf(conf, [0], [1])
print(conf)

#Random states
rg = nk.RandomEngine(seed=1234)
conf = np.zeros(h.size())
h.random_vals(conf, rg)
print(conf)
