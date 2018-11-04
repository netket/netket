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
import scipy.sparse as sparse

# #constructing a 1d lattice
gr=nk.Graph("Hypercube",L=20,Dimension=1)

# Hilbert space of spins from given graph
hi=nk.Hilbert(gr,Name="Spin",S=0.5)

#Hamiltonian
ha=nk.Hamiltonian(hi,Name="Ising",h=1.0)


v=np.ones(20)
#TODO define conversions between Vector types and python lists
mels=nk.VectorComplexDouble()
connectors=nk.VectorVectorInt()
newconfs=nk.VectorVectorDouble()
ha.FindConn(v,mels,connectors,newconfs)

for mel in mels:
    print(mel)

#scipy sparse diagonalization
print("Diagonalizing the Hamiltonian...")
sm=nk.SparseHamiltonianWrapper(ha).GetMatrix()

vals = sparse.linalg.eigs(sm, k=1,return_eigenvectors=False,which='SR')
print('Energy = ',vals[0].real)
print('Expected = ',-1.274549484318e+00*20)
