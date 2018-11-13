import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI

operators = {}

# Ising 1D
g_1 = nk.graph.Hypercube(L=20, ndim=1, pbc=True)
hi_1 = nk.hilbert.Spin(s=0.5, graph=g_1)
operators["Ising 1D"] = nk.operator.Ising(h=1.321, hilbert=hi_1)

# Heisenberg 1D
g_2 = nk.graph.Hypercube(L=20, ndim=1, pbc=True)
hi_2 = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g_1)
operators["Heisenberg 1D"] = nk.operator.Heisenberg(hilbert=hi_2)

# Bose Hubbard
g_3 = nk.graph.Hypercube(L=10, ndim=2, pbc=True)
hi_3 = nk.hilbert.Boson(n_max=3, n_bosons=23, graph=g_3)
operators["Bose Hubbard"] = nk.operator.BoseHubbard(U=4.0, hilbert=hi_3)

# Graph Hamiltonian
# TODO (jamesETsmith)

# Custom Hamiltonian
# TODO (jamesETsmith)
#sx = [[0,1],[1,0]]
#szsz = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
#sy = np.array([[0,1j],[-1j,0]])
#
#operators["Custom"] =


def test_produce_elements_in_hilbert():
    for name, ha in operators.items():
        hi = ha.GetHilbert()
        print(name, hi)
        assert (len(hi.local_states()) == hi.local_size())

        rstate = np.zeros(hi.size())
        rg = nk.RandomEngine(seed=1234)
        local_states = hi.local_states()

        for i in range(1000):

            hi.random_vals(rstate, rg)
            ha.GetConn(rstate)  #, mel, connectors, newconfs)
