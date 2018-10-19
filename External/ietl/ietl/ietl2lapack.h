/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2001-2010 by Prakash Dayal <prakash@comp-phys.org>,
 *                            Matthias Troyer <troyer@comp-phys.org>
 *
 * This software is part of the ALPS libraries, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 *
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

/* $Id: ietl2lapack.h,v 1.9 2003/12/05 09:24:01 tprosser Exp $ */

#ifndef EXTERNAL_IETL_IETL2LAPACK_H_
#define EXTERNAL_IETL_IETL2LAPACK_H_

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
#endif  // EXTERNAL_IETL_IETL2LAPACK_H_
