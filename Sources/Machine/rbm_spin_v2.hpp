// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef SOURCES_MACHINE_RBM_SPIN_V2_HPP
#define SOURCES_MACHINE_RBM_SPIN_V2_HPP

#include <cmath>
#include <memory>

#include <Eigen/Core>
#include <nonstd/optional.hpp>

#include "Hilbert/abstract_hilbert.hpp"

namespace netket {

// TODO: Remove me!
inline Complex SumLogCoshDumb(
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> input,
    Eigen::Ref<const Eigen::Matrix<Complex, Eigen::Dynamic, 1>> bias) {
  auto total = Complex{0.0, 0.0};
  for (auto i = Index{0}; i < input.size(); ++i) {
    total += std::log(std::cosh(input(i) + bias(i)));
  }
  return total;
}

class RbmSpinV2 {
 public:
  template <class T>
  using RowMatrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert, Index nhidden,
            Index alpha, bool usea, bool useb, Index const batch_size);

  Index Nvisible() const noexcept;
  Index Nhidden() const noexcept;
  Index Npar() const noexcept;
  Index BatchSize() const noexcept;

  Eigen::Ref<const Eigen::VectorXcd> LogVal(
      Eigen::Ref<const RowMatrix<double>> x);

  Eigen::Ref<Eigen::MatrixXcd> GetW() { return W_; }
  Eigen::Ref<Eigen::VectorXcd> GetA() { return a_.value(); }
  Eigen::Ref<Eigen::VectorXcd> GetB() { return b_.value(); }

 private:
  void ApplyBiasAndActivation();

  Eigen::MatrixXcd W_;             ///< weights
  nonstd::optional<VectorXcd> a_;  ///< visible units bias
  nonstd::optional<VectorXcd> b_;  ///< hidden units bias

  /// Caches
  RowMatrix<Complex> theta_;
  VectorXcd output_;

  /// Hilbert space
  std::shared_ptr<const AbstractHilbert> hilbert_;
};

}  // namespace netket

#endif  // SOURCES_MACHINE_RBM_SPIN_V2_HPP
