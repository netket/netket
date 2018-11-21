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
"""
  std::vector<std::vector<double>> sigmax = {{0, 1}, {1, 0}};
  std::vector<std::vector<double>> mszsz = {
      {1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};

  std::vector<std::vector<int>> edges = {
      {0, 1},   {1, 2},   {2, 3},   {3, 4},   {4, 5},   {5, 6},   {6, 7},
      {7, 8},   {8, 9},   {9, 10},  {10, 11}, {11, 12}, {12, 13}, {13, 14},
      {14, 15}, {15, 16}, {16, 17}, {17, 18}, {18, 19}, {19, 0}};

  pars.clear();

  pars["Graph"]["Edges"] = edges;
  pars["Hilbert"]["QuantumNumbers"] = {1, -1};
  pars["Hilbert"]["Size"] = edges.size();
  pars["Hamiltonian"]["Name"] = "Graph";
  pars["Hamiltonian"]["SiteOps"] = {sigmax};
  pars["Hamiltonian"]["BondOps"] = {mszsz};
  pars["Hamiltonian"]["BondOpColors"] = {0};
  input_tests.push_back(pars);
"""
sigmax = [[0, 1], [1, 0]]
mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
         [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
         [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]

g = nk.graph.CustomGraph(edges=edges)
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], graph=g)
ha = nk.operator.GraphHamiltonian(
    hi, siteops=[sigmax], bondops=[mszsz], bondops_colors=[0])
operators["Graph Hamiltonian"] = ha

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
                    assert (rs in local_states)
