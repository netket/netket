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

#ifndef NETKET_EXACT_DIAGONALIZATION_HPP
#define NETKET_EXACT_DIAGONALIZATION_HPP

#include <map>
#include <string>
#include <vector>

#include <ietl/lanczos.h>
#include <ietl/randomgenerator.h>

#include "Operator/MatrixWrapper/matrix_wrapper.hpp"
#include "Operator/hamiltonian.hpp"

namespace netket {

namespace eddetail {

using eigenvalues_t = std::vector<double>;
using eigenvectors_t =
    std::vector<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>>;

struct result_t {
  eigenvalues_t eigenvalues;
  int which_eigenvector;
  eigenvectors_t eigenvectors;
};

template <class matrix_t, class iter_t, class random_t>
result_t lanczos_run(const matrix_t& matrix, const random_t& random_gen,
                     iter_t& iter, int which_eigenvector = -1) {
  using complex = std::complex<double>;
  using vectorspace_t = ietl::vectorspace<complex>;
  using lanczos_t = ietl::lanczos<matrix_t, vectorspace_t>;

  size_t dimension = matrix.Dimension();
  vectorspace_t ietl_vecspace(dimension);
  lanczos_t lanczos(matrix, ietl_vecspace);
  lanczos.calculate_eigenvalues(iter, random_gen);
  result_t result = {lanczos.eigenvalues(), which_eigenvector,
                     eigenvectors_t()};
  if (iter.error_code() == 1)
    WarningMessage() << "Warning: Lanczos eigenvalue computation "
                     << "did NOT converge in " << iter.max_iterations()
                     << " steps!\n";
  if (which_eigenvector > -1) {
    result.eigenvectors.resize(1);
    ietl::Info<double> info;
    lanczos.eigenvectors(result.eigenvalues.begin() + which_eigenvector,
                         result.eigenvalues.begin() + which_eigenvector + 1,
                         result.eigenvectors.begin(), info, random_gen,
                         iter.max_iterations(), iter.max_iterations());
    if (info.error_info(0) != ietl::Info<double>::ok)
      WarningMessage() << "Warning: Lanczos eigenvector computation "
                       << "did NOT converge in " << iter.max_iterations()
                       << " steps!\n";
  }
  return result;
}
}  // namespace eddetail

eddetail::result_t lanczos_ed(const AbstractOperator& hamiltonian,
                              bool matrix_free = false, int first_n = 1,
                              int max_iter = 1000, int seed = 42,
                              double precision = 1e-14,
                              bool get_groundstate = false) {
  using normal_dist_t = std::uniform_real_distribution<double>;
  using random_t = ietl::random_generator<std::mt19937, normal_dist_t>;
  using iter_t = ietl::lanczos_iteration_nlowest<double>;
  normal_dist_t dist(-1., 1.);
  random_t random_gen(dist, seed);

  // Converge the first_n eigenvalues to precision
  iter_t iter(max_iter, first_n, sqrt(precision), sqrt(precision));
  eddetail::result_t results;

  int which_eigenvector = get_groundstate ? 0 : -1;

  if (matrix_free) {
    using matrix_t = DirectMatrixWrapper<AbstractOperator>;
    matrix_t matrix(hamiltonian);
    results =
        eddetail::lanczos_run(matrix, random_gen, iter, which_eigenvector);
  } else {  // computation using Sparse matrix
    using matrix_t = SparseMatrixWrapper<AbstractOperator>;
    matrix_t matrix(hamiltonian);
    results =
        eddetail::lanczos_run(matrix, random_gen, iter, which_eigenvector);
  }
  results.eigenvalues.resize(first_n);  // Keep only converged eigenvalues
  return results;
}

eddetail::result_t full_ed(const Hamiltonian& hamiltonian, int first_n = 1,
                           bool get_groundstate = false) {
  using eigen_solver_t =
      Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<std::complex<double>>>;

  SparseMatrixWrapper<Hamiltonian> matrix(hamiltonian);

  eddetail::result_t results;
  results.which_eigenvector = get_groundstate ? 0 : -1;

  eigen_solver_t eigen_solver;
  if (get_groundstate) {
    eigen_solver = matrix.ComputeEigendecomposition();
    results.eigenvectors.push_back(eigen_solver.eigenvectors().col(0));
  } else {
    eigen_solver = matrix.ComputeEigendecomposition(Eigen::EigenvaluesOnly);
  }
  auto eigen_evals = eigen_solver.eigenvalues();
  eigen_evals.conservativeResize(first_n);  // Keep only first_n eigenvalues
  results.eigenvalues = std::vector<double>(
      eigen_evals.data(),
      eigen_evals.data() + eigen_evals.rows() * eigen_evals.cols());
  return results;
}

void write_ed_results(const json& pars, const std::vector<double>& eigs,
                      const std::map<std::string, double>& observables) {
  std::string file_base = FieldVal(pars["GroundState"], "OutputFile");
  std::string file_name = file_base + std::string(".log");
  std::ofstream file_ed(file_name);
  json data;
  data["Eigenvalues"] = eigs;
  for (auto name_value : observables)
    data[name_value.first] = name_value.second;
  file_ed << data << std::endl;
  file_ed.close();
}

void get_ed_parameters(const json& pars, double& precision, int& n_eigenvalues,
                       int& random_seed, int& max_iter) {
  n_eigenvalues = FieldOrDefaultVal(pars["GroundState"], "NumEigenvalues", 1);
  precision = FieldOrDefaultVal(pars["GroundState"], "Precision", 1.0e-14);
  random_seed = FieldOrDefaultVal(pars["GroundState"], "RandomSeed", 42);
  max_iter = FieldOrDefaultVal(pars["GroundState"], "MaxIterations", 1000);
}

}  // namespace netket

#endif
