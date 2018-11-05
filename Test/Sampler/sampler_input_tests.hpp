// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fstream>
#include <string>
#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetSamplerInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

  // Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "MetropolisLocal"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Ising 1d with replicas
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "MetropolisLocalPt"}, {"Nreplicas", 4}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 6}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "MetropolisHamiltonian"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Ising 1d with replicas
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 6}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "MetropolisHamiltonianPt"}, {"Nreplicas", 4}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with symmetric machine
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 4}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpinSymm"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}},
          {"Sampler", {{"Name", "MetropolisLocal"}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 3;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 4}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}},
          {"Sampler", {{"Name", "MetropolisLocalPt"}, {"Nreplicas", 4}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 3;
  input_tests.push_back(pars);

  // Bose-Hubbard 1d with multi-val rbm
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 4}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmMultival"}, {"Alpha", 2.0}}},
          {"Hamiltonian", {{"Name", "BoseHubbard"}, {"U", 4.0}}},
          {"Sampler", {{"Name", "MetropolisLocalPt"}, {"Nreplicas", 4}}}};
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 3;
  input_tests.push_back(pars);
  //
  // Ising 1d with Custom Sampler
  std::vector<std::vector<double>> sx = {{0, 1}, {1, 0}};
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 6}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  pars["Sampler"]["MoveOperators"] = {sx, sx, sx, sx, sx, sx};
  pars["Sampler"]["ActingOn"] = {{0}, {1}, {2}, {3}, {4}, {5}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Ising 1D with CustomSampler two types of updates
  std::vector<std::vector<double>> spsm = {
      {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}};
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 4}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  pars["Sampler"]["MoveOperators"] = {sx, sx, sx, sx, spsm, spsm, spsm, spsm};
  pars["Sampler"]["ActingOn"] = {{0},    {1},    {2},    {3},
                                 {0, 1}, {1, 2}, {2, 3}, {3, 0}};
  input_tests.push_back(pars);

  // Ising 1d with Custom Sampler and replicas
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 4}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  pars["Sampler"]["MoveOperators"] = {sx, sx, sx, sx};
  pars["Sampler"]["ActingOn"] = {{0}, {1}, {2}, {3}};
  pars["Sampler"]["Nreplicas"] = 4;
  input_tests.push_back(pars);

  // Ising 1d with exact sampler
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"Sampler", {{"Name", "Exact"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  return input_tests;
}
