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
from mpi4py import MPI
from datetime import datetime
import numpy as np
import scipy.sparse.linalg as sparsediag

#Constructing a 1d lattice
g=nk.Hypercube(L=20,ndim=1)

#Hilbert space of spins from given graph
hi=nk.Spin(s=0.5,graph=g)

#Hamiltonian
ha=nk.Ising(h=1.0,hilbert=hi)

print("\n")
print("Diagonalizing the Hamiltonian with the internal NetKet solver...")

t1 = datetime.now()
ed_result=nk.LanczosEd(operator=ha,first_n=1,get_groundstate=False)
t2= datetime.now()

print("Elapsed time =",(t2-t1).total_seconds()," s\n")

#Scipy sparse diagonalization
print("Diagonalizing the Hamiltonian with scipy...")

t1 = datetime.now()
sm=nk.SparseMatrixWrapper(operator=ha).GetMatrix()
vals = sparsediag.eigs(sm, k=1,return_eigenvectors=False,which='SR')
t2 = datetime.now()

print("Elapsed time =",(t2-t1).total_seconds()," s\n")

print('Energy = ',ed_result.eigenvalues,vals[0].real)
print('Expected = ',-1.274549484318e+00*20)
