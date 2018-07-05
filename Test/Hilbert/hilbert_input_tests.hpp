
#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetHilbertInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

  // Spin 1/2
  pars = {{"Hilbert", {{"Name", "Spin"}, {"Nspins", 20}, {"S", 0.5}}}};
  input_tests.push_back(pars);

  // Spin 1/2 with total Sz
  pars = {{"Hilbert",
           {{"Name", "Spin"}, {"Nspins", 20}, {"S", 0.5}, {"TotalSz", 1.0}}}};
  input_tests.push_back(pars);

  // Spin 3
  pars = {{"Hilbert", {{"Name", "Spin"}, {"Nspins", 25}, {"S", 3}}}};
  input_tests.push_back(pars);

  // Boson
  pars = {{"Hilbert", {{"Name", "Boson"}, {"Nsites", 21}, {"Nmax", 5}}}};
  input_tests.push_back(pars);

  // Boson with total number
  pars = {{"Hilbert",
           {{"Name", "Boson"}, {"Nsites", 21}, {"Nmax", 5}, {"Nbosons", 11}}}};
  input_tests.push_back(pars);

  // Qubit
  pars = {{"Hilbert", {{"Name", "Qubit"}, {"Nqubits", 32}}}};
  input_tests.push_back(pars);

  // Custom Hilbert
  pars = {{"Hilbert", {{"QuantumNumbers", {-1232, 132, 0}}, {"Size", 34}}}};
  input_tests.push_back(pars);

  // Heisenberg 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}, {"TotalSz", 0}}}};
  input_tests.push_back(pars);

  // Bose Hubbard
  pars = {
      {"Graph",
       {{"Name", "Hypercube"}, {"L", 20}, {"Dimension", 1}, {"Pbc", true}}},
      {"Hamiltonian",
       {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 4}, {"Nbosons", 20}}}};
  input_tests.push_back(pars);

  // Small hilbert spaces
  // Spin 1/2
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

  return input_tests;
}
