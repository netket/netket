
#include "Json/json_helper.hpp"
#include <fstream>
#include <string>
#include <vector>

std::vector<json> GetSamplerInputs() {

  std::vector<json> input_tests;
  json pars;

  // Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "MetropolisLocal"}}}};
  input_tests.push_back(pars);

  // Ising 1d with replicas
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "MetropolisLocalPt"}, {"Nreplicas", 4}}}};
  input_tests.push_back(pars);

  // Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 6}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "MetropolisHamiltonian"}}}};
  input_tests.push_back(pars);

  // Ising 1d with replicas
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 6}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "MetropolisHamiltonianPt"}, {"Nreplicas", 4}}}};
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with symmetric machine
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 4}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 3}}},
          {"Sampler", {{"Name", "MetropolisLocal"}}}};
  input_tests.push_back(pars);

  // Bose-Hubbard 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 4}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 3}}},
          {"Sampler", {{"Name", "MetropolisLocalPt"}, {"Nreplicas", 4}}}};
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with multi-val rbm
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 4}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmMultival"}, {"Alpha", 2.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}, {"Nmax", 3}}},
          {"Sampler", {{"Name", "MetropolisLocalPt"}, {"Nreplicas", 4}}}};
  input_tests.push_back(pars);

  return input_tests;
}
