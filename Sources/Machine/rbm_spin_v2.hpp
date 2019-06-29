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
#include "Utils/log_cosh.hpp"

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
  template <class T>
  using M = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template <class T>
  using V = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  static constexpr auto D = Eigen::Dynamic;

  Eigen::Matrix<Complex, D, D, Eigen::ColMajor> W_;   ///< weights
  nonstd::optional<Eigen::Matrix<Complex, D, 1>> a_;  ///< visible units bias
  nonstd::optional<Eigen::Matrix<Complex, D, 1>> b_;  ///< hidden units bias

  /// Caches
  Eigen::Matrix<Complex, D, D, Eigen::RowMajor> theta_;
  Eigen::Matrix<Complex, D, 1> output_;

  /// Hilbert space
  std::shared_ptr<const AbstractHilbert> hilbert_;

 public:
  RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert, Index nhidden,
            Index alpha, bool usea, bool useb, Index const batch_size)
      : W_{}, a_{nonstd::nullopt}, b_{nonstd::nullopt}, theta_{}, output_{} {
    const auto nvisible = hilbert->Size();
    assert(nvisible >= 0 && "AbstractHilbert::Size is broken");
    if (nhidden < 0) {
      std::ostringstream msg;
      msg << "invalid number of hidden units: " << nhidden
          << "; expected a non-negative number";
      throw InvalidInputError{msg.str()};
    }
    if (alpha < 0) {
      std::ostringstream msg;
      msg << "invalid density of hidden units: " << alpha
          << "; expected a non-negative number";
      throw InvalidInputError{msg.str()};
    }
    if (nhidden > 0 && alpha > 0 && nhidden != alpha * nvisible) {
      std::ostringstream msg;
      msg << "number and density of hidden units are incompatible: " << nhidden
          << " != " << alpha << " * " << nvisible;
      throw InvalidInputError{msg.str()};
    }
    nhidden = std::max(nhidden, alpha * nvisible);

    W_.resize(nvisible, nhidden);
    if (usea) {
      a_.emplace(nvisible);
    }
    if (useb) {
      b_.emplace(nhidden);
    }

    theta_.resize(batch_size, nhidden);
    output_.resize(batch_size);
  }

  Index Nvisible() const noexcept { return W_.rows(); }
  Index Nhidden() const noexcept { return W_.cols(); }
  Index Npar() const noexcept {
    return W_.size() + (a_.has_value() ? a_->size() : 0) +
           (b_.has_value() ? b_->size() : 0);
  }
  Index BatchSize() const noexcept { return theta_.rows(); }

  Eigen::Ref<Eigen::Matrix<Complex, D, 1> const> LogVal(
      Eigen::Ref<const Eigen::MatrixXd> x) {
    if (x.rows() != BatchSize() || x.cols() != Nvisible()) {
      std::ostringstream msg;
      msg << "wrong shape: [" << x.rows() << ", " << x.cols() << "]; expected ["
          << BatchSize() << ", " << Nvisible() << "]\n";
      throw InvalidInputError{msg.str()};
    }
    if (a_.has_value()) {
      output_.noalias() = x * (*a_);
    } else {
      output_.setZero();
    }
    theta_.noalias() = x * W_;
    ApplyBiasAndActivation();
    return output_;
  }

  Eigen::Ref<Eigen::Matrix<Complex, D, D>> GetW() { return W_; }
  Eigen::Ref<Eigen::Matrix<Complex, D, 1>> GetA() { return a_.value(); }
  Eigen::Ref<Eigen::Matrix<Complex, D, 1>> GetB() { return b_.value(); }

  void ApplyBiasAndActivation() {
    if (b_.has_value()) {
#pragma omp parallel for
      for (auto j = Index{0}; j < BatchSize(); ++j) {
        output_(j) += SumLogCosh(theta_.row(j), (*b_));  // total;
      }
    } else {
#pragma omp parallel for
      for (auto j = Index{0}; j < BatchSize(); ++j) {
        output_(j) += SumLogCosh(theta_.row(j));
      }
    }
  }

  // PyObject *StateDict() const;
  // void StateDict(PyObject *dict);
};

}  // namespace netket

#endif  // SOURCES_MACHINE_RBM_SPIN_V2_HPP
