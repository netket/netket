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
  input_tests.push_back(pars);

  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 6}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", .5}}},
          {"GroundState", {{"Method", "Ed"}, {"OutputFile", "test"}}}};
  input_tests.push_back(pars);

  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.0}}},
          {"GroundState", {{"Method", "Ed"}, {"OutputFile", "test"}}}};
  input_tests.push_back(pars);

  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 8}, {"Dimension", 1}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", .5}}},
          {"GroundState", {{"Method", "Ed"}, {"OutputFile", "test"}}}};
  input_tests.push_back(pars);

  return input_tests;
}
