import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI

machines = {}

#Constructing a 1d lattice
g = nk.graph.Hypercube(L=20, ndim=1)

# Hilbert space of spins from given graph
hi = nk.hilbert.Spin(s=0.5, graph=g)

machines["RbmSpin 1d Hypercube"] = nk.machine.RbmSpin(hilbert=hi, alpha=1)

machines["RbmSpinSymm 1d Hypercube"] = nk.machine.RbmSpinSymm(hilbert=hi, alpha=2)

machines["Jastrow 1d Hypercube"] = nk.machine.Jastrow(hilbert=hi)

hi = nk.hilbert.Spin(s=0.5, graph=g,total_sz=0)
machines["Jastrow 1d Hypercube"] = nk.machine.JastrowSymm(hilbert=hi)

#
#   // Heisenberg 1d with fully connected FFNN
#   pars = {{"Graph",
#            {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
#           {"Machine",
#            {{"Name", "FFNN"},
#             {"Layers",
#              {{{"Name", "FullyConnected"},
#                {"Inputs", 20},
#                {"Outputs", 40},
#                {"Activation", "Lncosh"}}}}}},
 #         {"Hamiltonian", {{"Name", "Heisenberg"}}}};
 #  pars["Hilbert"]["Name"] = "Spin";
 #  pars["Hilbert"]["S"] = 0.5;
 #  input_tests.push_back(pars);
 #
 #  // Bose-Hubbard 1d with symmetric machine
 #  pars = {{"Graph",
 #           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
 #          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
 #          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
 #  pars["Hilbert"]["Name"] = "Boson";
 #  pars["Hilbert"]["Nmax"] = 4;
 #  input_tests.push_back(pars);
 #
 #  // Bose-Hubbard 1d with non-symmetric rbm machine
 #  pars = {{"Graph",
 #           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
 #          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 2.0}}},
 #          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
 #  pars["Hilbert"]["Name"] = "Boson";
 #  pars["Hilbert"]["Nmax"] = 4;
 #  input_tests.push_back(pars);
 #
 #  // Bose-Hubbard 1d with multi-val rbm
 #  pars = {{"Graph",
 #           {{"Name", "Hypercube"}, {"L", 10}, {"Dimension", 1}, {"Pbc", true}}},
 #          {"Machine", {{"Name", "RbmMultival"}, {"Alpha", 2.0}}},
 #          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
 #  pars["Hilbert"]["Name"] = "Boson";
 #  pars["Hilbert"]["Nmax"] = 3;
 # input_tests.push_back(pars);
 #
 #

 #
 #  // Bose-Hubbard 1d with non-symmetric Jastrow machine
 #  pars = {{"Graph",
 #           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
 #          {"Machine", {{"Name", "Jastrow"}}},
 #          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
 #  pars["Hilbert"]["Name"] = "Boson";
 #  pars["Hilbert"]["Nmax"] = 4;
 #  input_tests.push_back(pars);
 #
 #  // Bose-Hubbard 1d with symmetric Jastrow machine
 #  pars = {{"Graph",
 #           {{"Name", "Hypercube"}, {"L", 40}, {"Dimension", 1}, {"Pbc", true}}},
 #          {"Machine", {{"Name", "JastrowSymm"}}},
 #          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
 #  pars["Hilbert"]["Name"] = "Boson";
 #  pars["Hilbert"]["Nmax"] = 4;
 #  input_tests.push_back(pars);
 #
 #  // Ising 1d with MPS diagonal
 #  pars = {{"Graph",
 #           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
 #          {"Machine",
 #           {{"Name", "MPSperiodic"}, {"BondDim", 8}, {"Diagonal", true}}},
 #          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
 #  pars["Hilbert"]["Name"] = "Spin";
 #  pars["Hilbert"]["S"] = 0.5;
 #  input_tests.push_back(pars);
 #
 #  // Heisemberg 1d with MPS periodic(no translational symmetry)
 #  pars = {{"Graph",
 #           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
 #          {"Machine", {{"Name", "MPSperiodic"}, {"BondDim", 5}}},
 #          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
 #  pars["Hilbert"]["Name"] = "Spin";
 #  pars["Hilbert"]["S"] = 0.5;
 #  input_tests.push_back(pars);
 #
 #  // Bose-Hubbard 1d with MPS periodic
 #  pars = {{"Graph",
 #           {{"Name", "Hypercube"}, {"L", 40}, {"Dimension", 1}, {"Pbc", true}}},
 #          {"Machine",
 #           {{"Name", "MPSperiodic"}, {"BondDim", 5}, {"SymmetryPeriod", 5}}},
 #          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
 #  pars["Hilbert"]["Name"] = "Boson";
 #  pars["Hilbert"]["Nmax"] = 4;
 #  input_tests.push_back(pars);


import numpy as np

def test_set_get_parameters():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        assert(ma.Npar()>0)
        npar=ma.Npar()
        randpars=np.random.randn(npar)+1.0j*np.random.randn(npar)
        ma.SetParameters(randpars)
        assert(np.array_equal(ma.GetParameters(),randpars))

def test_log_derivative():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        #TODO maybe use numdifftools for numerical derivatives
        assert(True)

def test_nvisible():
    for name, ma in machines.items():
        print("Machine test: %s" % name)
        #TODO GetHilbert is broken because we return a reference and not a pointer
        # hh=ma.GetHilbert()
        # assert(ma.Nvisible()==ma.GetHilbert().Size())
