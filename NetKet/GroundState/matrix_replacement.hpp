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

#ifndef NETKET_MATRIXREPLACEMENT_HPP
#define NETKET_MATRIXREPLACEMENT_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <complex>
#include <iostream>
#include <unsupported/Eigen/IterativeSolvers>
#include "Utils/parallel_utils.hpp"

using Eigen::MatrixXcd;
using Eigen::MatrixXd;

namespace netket {
// Forward declaration
template <typename T>
class MatrixReplacement;
}  // namespace netket

namespace Eigen {
namespace internal {
// MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
template <typename T>
struct traits<netket::MatrixReplacement<T>>
    : public Eigen::internal::traits<Eigen::SparseMatrix<T>> {};
}  // namespace internal
}  // namespace Eigen

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
namespace netket {
template <typename T>
class MatrixReplacement
    : public Eigen::EigenBase<netket::MatrixReplacement<T>> {
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

 public:
  // Required typedefs, constants, and method:
  typedef Complex Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  Index rows() const { return mp_mat_.cols(); }
  Index cols() const { return mp_mat_.cols(); }
  template <typename Rhs>
  Eigen::Product<netket::MatrixReplacement<T>, Rhs, Eigen::AliasFreeProduct>
  operator*(const Eigen::MatrixBase<Rhs> &x) const {
    return Eigen::Product<netket::MatrixReplacement<T>, Rhs,
                          Eigen::AliasFreeProduct>(*this, x.derived());
  }
  // Custom API:
  MatrixReplacement() : shift_(0), scale_(1) {}
  void attachMatrix(const MatrixType &mat) { mp_mat_ = mat; }
  void setShift(double shift) { shift_ = shift; }
  MatrixType const &my_matrix() const { return mp_mat_; }
  double shift() const { return shift_; }
  void setScale(double scale) { scale_ = scale; }
  double getScale() const { return scale_; }

 private:
  MatrixType mp_mat_;
  double shift_;
  double scale_;
};
}  // namespace netket

// Implementation of MatrixReplacement * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {
template <typename T, typename Rhs>
struct generic_product_impl<netket::MatrixReplacement<T>, Rhs, SparseShape,
                            DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<
          netket::MatrixReplacement<T>, Rhs,
          generic_product_impl<netket::MatrixReplacement<T>, Rhs>> {
  typedef typename Product<netket::MatrixReplacement<T>, Rhs>::Scalar Scalar;
  template <typename Dest>
  static void scaleAndAddTo(Dest &dst, const netket::MatrixReplacement<T> &lhs,
                            const Rhs &rhs, const Scalar &alpha) {
    // This method should implement "dst += alpha * lhs * rhs" inplace,

    // auto vtilde = lhs.my_matrix() * rhs;
    // Eigen::VectorXcd res = lhs.my_matrix().adjoint() * vtilde;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> res =
        lhs.my_matrix() * rhs;
    netket::SumOnNodes(res);

    double nor = lhs.getScale();

    dst += alpha * (rhs * lhs.shift() + res * nor);
  }
};
}  // namespace internal
}  // namespace Eigen

#endif
