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

#include "catch.hpp"
#include "netket.hpp"
#include "Hamiltonian/MatrixWrapper/matrix_wrapper.hpp"
#include "Graph/graph.hpp"
#include "GroundState/exact_diagonalization.hpp"

#include "exact_diagonalization_input_tests.hpp"

TEST_CASE("Full / Lanczos ED give same ground state energy", "[ground state]") {
  auto input_tests = GetExactDiagonalizationInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string name = input_tests[i].dump();
    SECTION("Exact Diagonalization test (" +
            std::to_string(i) + ") on " + name) {
      netket::json pars = input_tests[i];
      netket::Graph graph(pars);
      netket::Hamiltonian hamiltonian(graph, pars);

      std::vector<double> eigs_lanczos = eigenvalues_lanczos(hamiltonian);
      std::vector<double> eigs_full = eigenvalues_full(hamiltonian);
      REQUIRE(std::abs(eigs_full[0] - eigs_lanczos[0]) < 1e-12);
    }
  }
}
