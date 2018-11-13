import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI

hilberts = {}

# Spin 1/2
hilberts["Spin 1/2"] = nk.hilbert.Spin(
    s=0.5, graph=nk.graph.Hypercube(L=20, ndim=1))

#Spin 1/2 with total Sz
hilberts["Spin 1/2 with total Sz"] = nk.hilbert.Spin(
    s=0.5, total_sz=1.0, graph=nk.graph.Hypercube(L=20, ndim=1))

# Spin 3
hilberts["Spin 3"] = nk.hilbert.Spin(
    s=3, graph=nk.graph.Hypercube(L=25, ndim=1))

# Boson
hilberts["Boson"] = nk.hilbert.Boson(
    n_max=5, graph=nk.graph.Hypercube(L=21, ndim=1))

# Boson with total number
hilberts["Bosons with total number"] = nk.hilbert.Boson(
    n_max=5, n_bosons=11, graph=nk.graph.Hypercube(L=21, ndim=1))

# Qubit
hilberts["Qubit"] = nk.hilbert.Qubit(graph=nk.graph.Hypercube(L=32, ndim=1))

# Custom Hilbert
hilberts["Custom Hilbert"] = nk.hilbert.CustomHilbert(
    local_states=[-1232, 132, 0], graph=nk.graph.Hypercube(L=34, ndim=1))

# Heisenberg 1d
g = nk.graph.Hypercube(L=20, ndim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, graph=g)
hilberts["Heisenberg 1d"] = nk.operator.Heisenberg(hilbert=hi).GetHilbert()

# Bose Hubbard
g = nk.graph.Hypercube(L=20, ndim=1, pbc=True)
hi = nk.hilbert.Boson(n_max=4, n_bosons=20, graph=g)
hilberts["Bose Hubbards"] = nk.operator.BoseHubbard(
    U=4.0, hilbert=hi).GetHilbert()
''' TODO (jamesETsmith)
#  // Small hilbert spaces
#  // Spin 1/2
  pars = {{"Hilbert", {{"Name", "Spin"}, {"Nspins", 10}, {"S", 0.5}}}};
  input_tests.push_back(pars);

  // Spin 3
  pars = {{"Hilbert", {{"Name", "Spin"}, {"Nspins", 4}, {"S", 3}}}};
  input_tests.push_back(pars);

  // Boson
  pars = {{"Hilbert", {{"Name", "Boson"}, {"Nsites", 5}, {"Nmax", 3}}}};
  input_tests.push_back(pars);

  // Qubit
  pars = {{"Hilbert", {{"Name", "Qubit"}, {"Nqubits", 11}}}};
  input_tests.push_back(pars);

  // Custom Hilbert
  pars = {{"Hilbert", {{"QuantumNumbers", {-1232, 132, 0}}, {"Size", 5}}}};
  input_tests.push_back(pars);

  // Heisenberg 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 9}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  input_tests.push_back(pars);

  // Bose Hubbard
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 2}, {"Dimension", 1}, {"Pbc", false}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 4}}}};
  input_tests.push_back(pars);
'''


def test_consistent_size():
    for name, hi in hilberts.items():
        print("Hilbert test: %s" % name)
        assert (hi.size() > 0)
        assert (hi.local_size() > 0)
        if hi.is_discrete():
            assert (len(hi.local_states()) == hi.local_size())
            for state in hi.local_states():
                assert (np.isfinite(state).all())


def test_random_states():
    for name, hi in hilberts.items():
        assert (hi.size() > 0)
        assert (hi.local_size() > 0)
        assert (len(hi.local_states()) == hi.local_size())

        if hi.is_discrete():
            rstate = np.zeros(hi.size())
            rg = nk.RandomEngine(seed=1234)
            local_states = hi.local_states()

            #print(type(local_states))
            for i in range(100):
                hi.random_vals(rstate, rg)
                for state in rstate:
                    assert (state in local_states)


""" TODO (jamesETsmith)
def test_mapping():

    for hi in hilberts:
        assert(hi.Size() > 0)
        assert(hi.LocalSize() > 0)

        if hi.Size() * np.log(hi.LocalSize() < np.log
"""
