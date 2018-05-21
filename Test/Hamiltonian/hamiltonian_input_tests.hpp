
#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetHamiltonianInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

  // Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.321}}}};
  input_tests.push_back(pars);

  // Heisenberg 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}, {"TotalSz", 0}}}};

  input_tests.push_back(pars);

  // Bose Hubbard
  pars = {
      {"Graph",
       {{"Name", "Hypercube"}, {"L", 10}, {"Dimension", 2}, {"Pbc", true}}},
      {"Hamiltonian",
       {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 9}, {"Nbosons", 23}}}};

  input_tests.push_back(pars);

  std::vector<std::vector<double>> sx = {{0, 1}, {1, 0}};
  std::vector<std::vector<double>> szsz = {
      {1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};
  std::complex<double> I(0, 1);
  std::vector<std::vector<std::complex<double>>> sy = {{0, I}, {-I, 0}};

  pars.clear();
  pars["Hilbert"]["QuantumNumbers"] = {1, -1};
  pars["Hilbert"]["Size"] = 10;
  pars["Hamiltonian"]["Operators"] = {sx, szsz, szsz, sx,   sy, sy,
                                      sy, szsz, sx,   szsz, sy, szsz};
  pars["Hamiltonian"]["ActingOn"] = {{0}, {0, 1}, {1, 0}, {1},    {2}, {3},
                                     {4}, {4, 5}, {5},    {6, 8}, {9}, {7, 0}};

  input_tests.push_back(pars);
  return input_tests;
}
