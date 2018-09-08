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

#include <vector>

#include <ietl/lanczos.h>
#include <ietl/randomgenerator.h>

#include "Hamiltonian/MatrixWrapper/matrix_wrapper.hpp"

namespace netket {

std::vector<double> eigenvalues_lanczos(const Hamiltonian &hamiltonian,
                                        int first_n = 1) {
  using complex = std::complex<double>;
  using normal_dist_t = std::uniform_real_distribution<double>;
  using matrix_t = Eigen::SparseMatrix<complex>;
  using vectorspace_t = ietl::vectorspace<complex>;
  using iter_t = ietl::lanczos_iteration_nlowest<double>;
  using lanczos_t = ietl::lanczos<matrix_t, vectorspace_t>;

  // TODO: make these parameters accessible to user
  int max_iter = 1000;
  int seed = 42;
  double precision = 1e-8;

  SparseMatrixWrapper<Hamiltonian> matrix(hamiltonian);
  const matrix_t &numerical_matrix = matrix.GetMatrix();

  size_t dimension = numerical_matrix.rows();
  vectorspace_t ietl_vecspace(dimension);

  int num_eigenvalue_to_converge = first_n;

  normal_dist_t dist(-1., 1.);
  ietl::random_generator<std::mt19937, normal_dist_t> random_gen(dist, seed);
  iter_t iter(max_iter, num_eigenvalue_to_converge, precision, precision);
  lanczos_t lanczos(numerical_matrix, ietl_vecspace);

  lanczos.calculate_eigenvalues(iter, random_gen);

  std::vector<double> eigs_lanczos = lanczos.eigenvalues();
  eigs_lanczos.resize(first_n);

  return eigs_lanczos;
}

std::vector<double> eigenvalues_full(const Hamiltonian &hamiltonian,
                                     int first_n = 1) {
  SparseMatrixWrapper<Hamiltonian> matrix(hamiltonian);
  auto ed = matrix.ComputeEigendecomposition(Eigen::EigenvaluesOnly);
  auto eigs = ed.eigenvalues();
  eigs.conservativeResize(first_n);

  return std::vector<double>(eigs.data(),
                             eigs.data() + eigs.rows() * eigs.cols());
}
}  // namespace netket

#endif
