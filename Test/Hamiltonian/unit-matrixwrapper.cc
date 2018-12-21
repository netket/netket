// Copyright 2018 Damian Hofmann - All Rights Reserved.
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

#include "Graph/graph.hpp"
#include "Operator/MatrixWrapper/dense_matrix_wrapper.hpp"
#include "Operator/MatrixWrapper/direct_matrix_wrapper.hpp"
#include "Operator/MatrixWrapper/sparse_matrix_wrapper.hpp"
#include "Operator/hamiltonian.hpp"
#include "Operator/observable.hpp"

#include "../Observable/observable_input_tests.hpp"

std::vector<netket::json> GetHamiltonianInputs() {
  std::vector<netket::json> input_tests;
  netket::json pars;

  // Ising 1d
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 5}, {"Dimension", 1}, {"Pbc", false}}},
          {"Machine", {{"Name", "RbmSpin"}, {"Alpha", 1.0}}},
          {"Hamiltonian", {{"Name", "Ising"}, {"h", 1.321}}}};
  pars["Hilbert"]["QuantumNumbers"] = {-1, 1};
  input_tests.push_back(pars);

  // Heisenberg 1d
  pars.clear();
  pars = {{"Graph",
           {{"Name", "Hypercube"}, {"L", 3}, {"Dimension", 2}, {"Pbc", true}}},
          {"Hamiltonian", {{"Name", "Heisenberg"}}}};
  pars["Hilbert"]["Name"] = "Spin";
  pars["Hilbert"]["S"] = 0.5;
  pars["Hilbert"]["TotalSz"] = 0.;
  input_tests.push_back(pars);

  // Bose Hubbard
  pars.clear();
  pars["Graph"]["Name"] = "Hypercube";
  pars["Graph"]["L"] = 2;
  pars["Graph"]["Dimension"] = 2;
  pars["Graph"]["Pbc"] = false;
  pars["Hilbert"]["Name"] = "Boson";
  pars["Hilbert"]["Nmax"] = 2;
  pars["Hamiltonian"]["Name"] = "BoseHubbard";
  pars["Hamiltonian"]["U"] = 4.0;
  input_tests.push_back(pars);

  std::vector<std::vector<double>> sx = {{0, 1}, {1, 0}};
  std::vector<std::vector<double>> szsz = {
      {1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};

  Complex Iu(0, 1);
  std::vector<std::vector<Complex>> sy = {{0, Iu}, {-Iu, 0}};

  pars.clear();
  pars["Hilbert"]["QuantumNumbers"] = {1, -1};
  pars["Graph"]["Name"] = "Custom";
  pars["Graph"]["Size"] = 10;
  pars["Hamiltonian"]["Operators"] = {sx, szsz, szsz, sx,   sy, sy,
                                      sy, szsz, sx,   szsz, sy, szsz};
  pars["Hamiltonian"]["ActingOn"] = {{0}, {0, 1}, {1, 0}, {1},    {2}, {3},
                                     {4}, {4, 5}, {5},    {6, 8}, {9}, {7, 0}};

  input_tests.push_back(pars);
  return input_tests;
}

TEST_CASE("SparseMatrixWrapper for Hamiltonian is Hermitian",
          "[matrix-wrapper]") {
  auto input_tests = GetHamiltonianInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Hamiltonian test (" + std::to_string(it) + ") on " +
            input_tests[it]["Hamiltonian"].dump()) {
      auto pars = input_tests[it];
      netket::Graph graph(pars);
      netket::Hilbert hilbert(graph, pars);

      netket::Hamiltonian hamiltonian(hilbert, pars);

      netket::SparseMatrixWrapper<> hmat(hamiltonian);

      const auto& matrix = hmat.GetMatrix();
      REQUIRE(matrix.isApprox(matrix.adjoint()));
    }
  }
}

TEST_CASE("DenseMatrixWrapper for Hamiltonian is Hermitian",
          "[matrix-wrapper]") {
  auto input_tests = GetHamiltonianInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Hamiltonian test (" + std::to_string(it) + ") on " +
            input_tests[it]["Hamiltonian"].dump()) {
      auto pars = input_tests[it];
      netket::Graph graph(pars);
      netket::Hilbert hilbert(graph, pars);

      netket::Hamiltonian hamiltonian(hilbert, pars);

      netket::DenseMatrixWrapper<> hmat(hamiltonian);

      const auto& matrix = hmat.GetMatrix();
      REQUIRE(matrix.isApprox(matrix.adjoint()));
    }
  }
}

TEST_CASE("DirectMatrixWrapper gives same results as SparseMatrixWrapper",
          "[matrix-wrapper]") {
  auto input_tests = GetHamiltonianInputs();
  std::size_t ntests = input_tests.size();

  for (std::size_t it = 0; it < ntests; it++) {
    SECTION("Hamiltonian test (" + std::to_string(it) + ") on " +
            input_tests[it]["Hamiltonian"].dump()) {
      auto pars = input_tests[it];
      netket::Graph graph(pars);
      netket::Hilbert hilbert(graph, pars);

      netket::Hamiltonian hamiltonian(hilbert, pars);

      netket::DirectMatrixWrapper<> direct(hamiltonian);
      netket::SparseMatrixWrapper<> sparse(hamiltonian);

      Eigen::VectorXcd basis(direct.Dimension());
      Eigen::VectorXcd direct_result(direct.Dimension());
      Eigen::VectorXcd sparse_result(direct.Dimension());
      for (int i = 0; i < direct.Dimension(); i++) {
        basis.setZero();
        basis(i) = 1.0;
        direct_result = direct.Apply(basis);
        sparse_result = sparse.Apply(basis);

        INFO("i=" << i);
        REQUIRE(direct_result.isApprox(sparse_result));
      }
    }
  }
}

TEST_CASE("MatrixWrappers compute correct eigenvalues", "[matrix-wrapper]") {
  netket::json pars;

  pars["Hilbert"]["QuantumNumbers"] = {-1, 1};
  pars["Hilbert"]["Size"] = 1;
  pars["Graph"]["Name"] = "Custom";
  pars["Graph"]["Size"] = 1;

  netket::json observable_pars;
  observable_pars["ActingOn"] = {{0}};
  observable_pars["Operators"] = {{{-1, 2}, {2, 1}}};
  observable_pars["Name"] = "O1";

  netket::Hilbert hilbert(pars);
  netket::Observable obs(hilbert, observable_pars);

  // check whether the correct eigenvalues are computed
  {
    netket::DenseMatrixWrapper<> dense(obs);

    auto ed = dense.ComputeEigendecomposition();
    auto eigs = ed.eigenvalues();
    std::sort(eigs.data(), eigs.data() + eigs.size());

    const double sqrt5 = std::sqrt(5);
    CHECK(eigs(0) == Approx(-sqrt5));
    CHECK(eigs(1) == Approx(sqrt5));
  }
  {
    netket::SparseMatrixWrapper<> sparse(obs);

    auto ed = sparse.ComputeEigendecomposition();
    auto eigs = ed.eigenvalues();
    std::sort(eigs.data(), eigs.data() + eigs.size());

    const double sqrt5 = std::sqrt(5);
    CHECK(eigs(0) == Approx(-sqrt5));
    CHECK(eigs(1) == Approx(sqrt5));
  }
}
