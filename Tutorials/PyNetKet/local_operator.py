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
import numpy as np
from mpi4py import MPI
from datetime import datetime
import scipy.sparse as sparse

#Constructing a 1d lattice
g = nk.graph.Hypercube(L=10, ndim=1)

#Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

Z = [[1, 0], [0, -1]]
X = [[0, 1], [1, 0]]
Y = [[0, 1.0j], [-1.0j, 0.0]]

#Local Operator
#here heisenberg term \vec{sigma}_0 \cdot \vec{sigma}_1
#showcasing automatic simplifications and tensor products
o1 = nk.operator.LocalOperator(hi, X, [0]) * (nk.operator.LocalOperator(
    hi, X, [1]))
# o1+=(nk.LocalOperator(hi,Y,[0])*nk.LocalOperator(hi,Y,[1]))
# o1+=(nk.LocalOperator(hi,Z,[0])*nk.LocalOperator(hi,Z,[1]))

for m in o1.LocalMatrices():
    print(m, '\n')

#Find the connected elements of the operator
v = np.ones(10)
(mel, connectors, newconfs) = o1.GetConn(v)
print(connectors)
