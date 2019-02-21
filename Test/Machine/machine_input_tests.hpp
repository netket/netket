
#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetMachineInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

  // Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Heisenberg 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 2}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Heisenberg 1d with fully connected FFNN
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine",
           {{"Name", "FFNN"},
            {"Layers",
             {{{"Name", "FullyConnected"},
               {"Inputs", 20},
               {"Outputs", 40},
               {"Activation", "Lncosh"}}}}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with symmetric machine
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 4;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with non-symmetric rbm machine
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 2.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 4;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with multi-val rbm
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 10}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmMultival"}, {"Alpha", 2.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 3;
  input_tests.push_back(pars);

  // Ising 1d with jastrow
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "Jastrow"}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 2.0}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Heisemberg 1d with symmetric jastrow
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "JastrowSymm"}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  pars["Hilbert"]["TotalSz"] = 0.;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with non-symmetric Jastrow machine
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "Jastrow"}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 4;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with symmetric Jastrow machine
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 40}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "JastrowSymm"}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 4;
  input_tests.push_back(pars);

  // Ising 1d with MPS diagonal
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine",
           {{"Name", "MPSperiodic"}, {"BondDim", 8}, {"Diagonal", true}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Heisemberg 1d with MPS periodic(no translational symmetry)
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "MPSperiodic"}, {"BondDim", 5}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with MPS periodic
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 40}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine",
           {{"Name", "MPSperiodic"}, {"BondDim", 5}, {"SymmetryPeriod", 5}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 4;
  input_tests.push_back(pars);

  return input_tests;
}
