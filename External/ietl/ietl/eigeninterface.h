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

#ifndef EXTERNAL_IETL_IETL_EIGENINTERFACE_H_
#define EXTERNAL_IETL_IETL_EIGENINTERFACE_H_

#include <ietl/complex.h>
#include <Eigen/SparseCore>

namespace ietl {

template <class Gen>
void generate(Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>& c,
              Gen& gen) {
  for (int i = 0; i < c.rows(); ++i)
    c(i, 0) = std::complex<double>(gen(), gen());
}

template <class Gen>
void generate(Eigen::Matrix<double, Eigen::Dynamic, 1>& c, Gen& gen) {
  for (int i = 0; i < c.rows(); ++i) c(i, 0) = static_cast<double>(gen());
}

template <class TCoeffs>
void mult(const Eigen::SparseMatrix<TCoeffs>& a,
          const Eigen::Matrix<TCoeffs, Eigen::Dynamic, 1>& x,
          Eigen::Matrix<TCoeffs, Eigen::Dynamic, 1>& y) {
  y = a * x;
}

template <class matrix_t, class TCoeffs>
void mult(const matrix_t& a, const Eigen::Matrix<TCoeffs, Eigen::Dynamic, 1>& x,
          Eigen::Matrix<TCoeffs, Eigen::Dynamic, 1>& y) {
  y = a.Apply(x);
}

template <class TCoeffs>
TCoeffs dot(const Eigen::Matrix<TCoeffs, Eigen::Dynamic, 1>& x,
            const Eigen::Matrix<TCoeffs, Eigen::Dynamic, 1>& y) {
  return y.dot(x);
}

template <class TCoeffs>
typename real_type<TCoeffs>::type two_norm(
    const Eigen::Matrix<TCoeffs, Eigen::Dynamic, 1>& c) {
  return std::sqrt(ietl::real<typename real_type<TCoeffs>::type>(c.dot(c)));
}
}  // namespace ietl

#endif  // EXTERNAL_IETL_IETL_EIGENINTERFACE_H_
