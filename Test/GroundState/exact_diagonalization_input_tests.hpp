// Copyright 2018 Alexander Wietek - All Rights Reserved.
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

#include <vector>
#include "Utils/json_utils.hpp"

std::vector<netket::json> GetExactDiagonalizationInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

  // Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 6}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"GroundState", {{"Method", "Ed"}, {"OutputFile", "test"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 6}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", .5}}},
          {"GroundState", {{"Method", "Ed"}, {"OutputFile", "test"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"GroundState", {{"Method", "Ed"}, {"OutputFile", "test"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", .5}}},
          {"GroundState", {{"Method", "Ed"}, {"OutputFile", "test"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  input_tests.push_back(pars);

  // Complex hamiltonian
  std::vector<std::vector<double>> sx = {{0, 1}, {1, 0}};
  std::vector<std::vector<double>> szsz = {
      {1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};

  Complex Iu(0, 1);
  std::vector<std::vector<Complex>> sy = {{0, Iu}, {-Iu, 0}};

  pars.clear();

  pars["Hilbert"]["QuantumNumbers"] = {1, -1};
  pars["Hilbert"]["Size"] = 8;
  pars["Hamiltonian"]["Operators"] = {sx, szsz, szsz, sx, sy,
                                      sy, sy,   szsz, sx, szsz};
  pars["Hamiltonian"]["ActingOn"] = {{0}, {0, 1}, {1, 0}, {1}, {2},
                                     {3}, {4},    {4, 5}, {5}, {7, 0}};

  input_tests.push_back(pars);

  return input_tests;
}
