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

#include "Graph/graph.hpp"
#include "GroundState/exact_diagonalization.hpp"
#include "Operator/MatrixWrapper/matrix_wrapper.hpp"
#include "catch.hpp"
#include "netket.hpp"

#include "exact_diagonalization_input_tests.hpp"

TEST_CASE("Full / Lanczos ED give same ground state energy", "[ground state]") {
  auto input_tests = GetExactDiagonalizationInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t i = 0; i < ntests; i++) {
    std::string name = input_tests[i].dump();
    SECTION("Exact Diagonalization test (" + std::to_string(i) + ") on " +
            name) {
      netket::json pars = input_tests[i];
      netket::Graph graph(pars);
      netket::Hilbert hilbert(graph, pars);

      netket::Hamiltonian hamiltonian(hilbert, pars);

      // Check whether full and sparse ED yield same result
      auto result_lanczos = lanczos_ed(hamiltonian);
      auto result_full = full_ed(hamiltonian);
      REQUIRE(std::abs(result_full.eigenvalues[0] -
                       result_lanczos.eigenvalues[0]) < 1e-12);

      // Check whether ground state has lowest eigenvalue energy
      auto mat = netket::SparseMatrixWrapper<>(hamiltonian);
      result_lanczos = lanczos_ed(hamiltonian, false, 1, 1000, 42, 1e-12, true);
      const auto state_lanczos = result_lanczos.eigenvectors[0];
      auto mean_variance_lanczos = mat.MeanVariance(state_lanczos);
      REQUIRE(std::abs(mean_variance_lanczos[0] -
                       result_lanczos.eigenvalues[0]) < 1e-10);
      REQUIRE(std::abs(mean_variance_lanczos[1]) < 1e-10);

      result_full = full_ed(hamiltonian, 1, true);
      const auto state_full = result_full.eigenvectors[0];
      auto mean_variance_full = mat.MeanVariance(state_full);
      REQUIRE(std::abs(mean_variance_full[0] - result_full.eigenvalues[0]) <
              1e-10);
      REQUIRE(std::abs(mean_variance_full[1]) < 1e-10);
    }
  }
}
