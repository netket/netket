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

/* author: Alexander Wietek, 2018/10/22 */

#ifndef EXTERNAL_IETL_IETL_IETL2LAPACK_H_
#define EXTERNAL_IETL_IETL_IETL2LAPACK_H_

#include <cassert>
#include <cstdlib>
#include <stdexcept>

#include <algorithm>
#include <complex>
#include <vector>

#include <Eigen/Eigenvalues>

namespace ietl {
template <class T>
T* get_data(const std::vector<T>& v) {
  if (v.empty())
    return 0;
  else
    return const_cast<T*>(&v[0]);
}
}  // namespace ietl

namespace ietl2lapack {
typedef int fortran_int_t;

namespace ietl_lapack_detail {

template <class vector_t>
Eigen::Matrix<typename vector_t::value_type, Eigen::Dynamic, Eigen::Dynamic>
fill_tmatrix_eigen(const vector_t& alpha, const vector_t& beta) {
  using real_t = typename vector_t::value_type;
  using matrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;

  size_t size = alpha.size();
  assert(beta.size() == size);

  // Fill T-matrix (could probably be done better)
  matrix_t Tmat = matrix_t::Zero(size, size);
  for (size_t i = 0; i < size; ++i) {
    Tmat(i, i) = alpha[i];
    if (i != size - 1) {
      Tmat(i, i + 1) = beta[i];
      Tmat(i + 1, i) = beta[i];
    }
  }

  return Tmat;
}

}  // namespace ietl_lapack_detail

template <class vector_t>
fortran_int_t stev(const vector_t& alpha, const vector_t& beta, vector_t& eval,
                   fortran_int_t /*n*/) {
  using real_t = typename vector_t::value_type;
  using matrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;

  matrix_t Tmat = ietl_lapack_detail::fill_tmatrix_eigen(alpha, beta);

  Eigen::SelfAdjointEigenSolver<matrix_t> eigensolver;
  eigensolver.compute(Tmat);
  auto eigenvalues = eigensolver.eigenvalues();

  eval.resize(eigenvalues.rows() * eigenvalues.cols());
  std::copy(eigenvalues.data(),
            eigenvalues.data() + eigenvalues.rows() * eigenvalues.cols(),
            eval.begin());

  return eigensolver.info();
}

template <class vector_t, class FortranMatrix>
int stev(const vector_t& alpha, const vector_t& beta, vector_t& eval,
         FortranMatrix& z, fortran_int_t /*n*/) {
  using real_t = typename vector_t::value_type;
  using matrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;

  matrix_t Tmat = ietl_lapack_detail::fill_tmatrix_eigen(alpha, beta);

  Eigen::SelfAdjointEigenSolver<matrix_t> eigensolver;
  eigensolver.compute(Tmat);
  auto eigenvalues = eigensolver.eigenvalues();
  auto eigenvectors = eigensolver.eigenvectors();
  eval.resize(eigenvalues.size());
  std::copy(eigenvalues.data(),
            eigenvalues.data() + eigenvalues.rows() * eigenvalues.cols(),
            eval.begin());
  z.resize(eigenvectors.rows(), eigenvectors.cols());
  std::copy(eigenvectors.data(),
            eigenvectors.data() + eigenvectors.rows() * eigenvectors.cols(),
            z.data());

  return eigensolver.info();
}
}  // namespace ietl2lapack
#endif  // EXTERNAL_IETL_IETL_IETL2LAPACK_H_
