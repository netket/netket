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
#include "Operator/operator.hpp"
#include "common_types.hpp"

namespace netket {

namespace eddetail {

using eigenvalues_t = std::vector<double>;
using eigenvectors_t = std::vector<Eigen::Matrix<Complex, Eigen::Dynamic, 1>>;

class result_t {
 public:
  result_t(eigenvalues_t eigenvalues, eigenvectors_t eigenvectors)
      : eigenvalues_(std::move(eigenvalues)),
        eigenvectors_(std::move(eigenvectors)) {}

  eigenvalues_t& eigenvalues() { return eigenvalues_; }
  eigenvectors_t& eigenvectors() { return eigenvectors_; }

  Complex mean(AbstractOperator& op, int which = 0) {
    assert(which >= 0 &&
           static_cast<std::size_t>(which) < eigenvectors_.size());
    DirectMatrixWrapper<> op_mat(op);
    return op_mat.Mean(eigenvectors_[which]);
  }

 private:
  eigenvalues_t eigenvalues_;
  eigenvectors_t eigenvectors_;
};

template <class matrix_t, class iter_t, class random_t>
result_t lanczos_run(const matrix_t& matrix, const random_t& random_gen,
                     iter_t& iter, int n_eigenvectors) {
  using vectorspace_t = ietl::vectorspace<Complex>;
  using lanczos_t = ietl::lanczos<matrix_t, vectorspace_t>;

  size_t dimension = matrix.Dimension();
  vectorspace_t ietl_vecspace(dimension);
  lanczos_t lanczos(matrix, ietl_vecspace);
  lanczos.calculate_eigenvalues(iter, random_gen);

  eigenvectors_t eigenvectors;
  if (iter.error_code() == 1)
    WarningMessage() << "Warning: Lanczos eigenvalue computation "
                     << "did NOT converge in " << iter.max_iterations()
                     << " steps!\n";
  if (n_eigenvectors > 0) {
    eigenvectors.resize(n_eigenvectors);
    ietl::Info<double> info;
    lanczos.eigenvectors(lanczos.eigenvalues().begin(),
                         lanczos.eigenvalues().begin() + n_eigenvectors,
                         eigenvectors.begin(), info, random_gen,
                         iter.max_iterations(), iter.max_iterations());
    if (info.error_info(0) != ietl::Info<double>::ok)
      WarningMessage() << "Warning: Lanczos eigenvector computation "
                       << "did NOT converge in " << iter.max_iterations()
                       << " steps!\n";
  }
  result_t result(lanczos.eigenvalues(), std::move(eigenvectors));
  return result;
}
}  // namespace eddetail

eddetail::result_t lanczos_ed(const AbstractOperator& op,
                              bool matrix_free = false, int first_n = 1,
                              int max_iter = 1000, int seed = 42,
                              double precision = 1e-14,
                              bool compute_eigenvectors = false) {
  using normal_dist_t = std::uniform_real_distribution<double>;
  using random_t = ietl::random_generator<std::mt19937, normal_dist_t>;
  using iter_t = ietl::lanczos_iteration_nlowest<double>;
  normal_dist_t dist(-1., 1.);
  random_t random_gen(dist, seed);

  // Converge the first_n eigenvalues to precision
  iter_t iter(max_iter, first_n, sqrt(precision), sqrt(precision));

  int n_eigenvectors = compute_eigenvectors ? first_n : 0;

  if (matrix_free) {
    DirectMatrixWrapper<> matrix(op);
    eddetail::result_t results =
        eddetail::lanczos_run(matrix, random_gen, iter, n_eigenvectors);
    results.eigenvalues().resize(first_n);  // Keep only converged eigenvalues
    return results;
  } else {  // computation using Sparse matrix
    SparseMatrixWrapper<> matrix(op);
    eddetail::result_t results =
        eddetail::lanczos_run(matrix, random_gen, iter, n_eigenvectors);
    results.eigenvalues().resize(first_n);  // Keep only converged eigenvalues
    return results;
  }
}

eddetail::result_t full_ed(const AbstractOperator& op, int first_n = 1,
                           bool compute_eigenvectors = false) {
  using eigen_solver_t =
      Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<Complex>>;

  SparseMatrixWrapper<> matrix(op);

  eigen_solver_t eigen_solver;
  eddetail::eigenvectors_t eigenvectors;
  if (compute_eigenvectors) {
    eigen_solver = matrix.ComputeEigendecomposition();
    for (int i = 0; i < first_n; ++i)
      eigenvectors.push_back(eigen_solver.eigenvectors().col(i));
  } else {
    eigen_solver = matrix.ComputeEigendecomposition(Eigen::EigenvaluesOnly);
  }
  auto eigen_evals = eigen_solver.eigenvalues();
  eigen_evals.conservativeResize(first_n);  // Keep only first_n eigenvalues
  eddetail::eigenvalues_t eigenvalues = std::vector<double>(
      eigen_evals.data(),
      eigen_evals.data() + eigen_evals.rows() * eigen_evals.cols());

  eddetail::result_t results(std::move(eigenvalues), std::move(eigenvectors));
  return results;
}

}  // namespace netket

#endif
