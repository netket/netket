import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI

operators = {}

# Ising 1D
g = nk.graph.Hypercube(length=20, ndim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, graph=g)
operators["Ising 1D"] = nk.operator.Ising(h=1.321, hilbert=hi)

# Heisenberg 1D
g = nk.graph.Hypercube(length=20, ndim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
operators["Heisenberg 1D"] = nk.operator.Heisenberg(hilbert=hi)

# Bose Hubbard
g = nk.graph.Hypercube(length=3, ndim=2, pbc=True)
hi = nk.hilbert.Boson(n_max=3, n_bosons=6, graph=g)
operators["Bose Hubbard"] = nk.operator.BoseHubbard(U=4.0, hilbert=hi)

# Graph Hamiltonian
# TODO (jamesETsmith)

# Custom Hamiltonian
# TODO (jamesETsmith)
#sx = [[0,1],[1,0]]
#szsz = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
#sy = np.array([[0,1j],[-1j,0]])
#
# operators["Custom"] =
rg = nk.utils.RandomEngine(seed=1234)


def test_produce_elements_in_hilbert():
    for name, ha in operators.items():
        hi = ha.get_hilbert()
        print(name, hi)
        assert (len(hi.local_states()) == hi.local_size())

        rstate = np.zeros(hi.size())

        local_states = hi.local_states()

        for i in range(1000):
            hi.random_vals(rstate, rg)
            conns = ha.get_conn(rstate)

            for connector, newconf in zip(conns[1], conns[2]):
                rstatet = np.array(rstate)
                hi.update_conf(rstatet, connector, newconf)

                for rs in rstatet:
                    assert(rs in local_states)


def test_local_operator_ops():
    sx = [[0, 1], [1, 0]]

    sy = np.array([[0, 1j], [-1j, 0]]).tolist()
    sz = [[1, 0], [0, -1]]

    g = nk.graph.CustomGraph(edges=[[i, i + 1] for i in range(20)])
    hi = nk.hilbert.CustomHilbert(local_states=[1, -1], graph=g)

    sx_hat = nk.operator.LocalOperator(hi, [sx] * 3, [[0], [1], [5]])
    sy_hat = nk.operator.LocalOperator(hi, [sy] * 4, [[2], [3], [4], [9]])
    szsz_hat = nk.operator.LocalOperator(
        hi, sz, [0]) * nk.operator.LocalOperator(hi, sz, [1])
    szsz_hat += nk.operator.LocalOperator(hi, sz,
                                          [4]) * nk.operator.LocalOperator(hi, sz, [5])
    szsz_hat += nk.operator.LocalOperator(hi, sz,
                                          [6]) * nk.operator.LocalOperator(hi, sz, [8])
    szsz_hat += nk.operator.LocalOperator(hi, sz,
                                          [7]) * nk.operator.LocalOperator(hi, sz, [0])

    operators["Custom Hamiltonian"] = sx_hat + sy_hat
