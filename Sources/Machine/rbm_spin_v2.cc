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

#include "Machine/rbm_spin_v2.hpp"
#include "Utils/log_cosh.hpp"

namespace netket {

RbmSpinV2::RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert,
                     Index nhidden, Index alpha, bool usea, bool useb,
                     Index const batch_size)
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

Index RbmSpinV2::Nvisible() const noexcept { return W_.rows(); }
Index RbmSpinV2::Nhidden() const noexcept { return W_.cols(); }
Index RbmSpinV2::Npar() const noexcept {
  return W_.size() + (a_.has_value() ? a_->size() : 0) +
         (b_.has_value() ? b_->size() : 0);
}
Index RbmSpinV2::BatchSize() const noexcept { return theta_.rows(); }

void RbmSpinV2::BatchSize(Index batch_size) {
  if (batch_size <= 0) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  if (batch_size != BatchSize()) {
    theta_.resize(batch_size, theta_.cols());
    output_.resize(batch_size);
  }
}

Eigen::Ref<const Eigen::VectorXcd> RbmSpinV2::LogVal(
    Eigen::Ref<const RowMatrix<double>> x) {
  if (x.cols() != Nvisible()) {
    std::ostringstream msg;
    msg << "wrong shape: [" << x.rows() << ", " << x.cols() << "]; expected [?"
        << ", " << Nvisible() << "]";
    throw InvalidInputError{msg.str()};
  }
  BatchSize(x.rows());
  if (a_.has_value()) {
    output_.noalias() = x * (*a_);
  } else {
    output_.setZero();
  }
  theta_.noalias() = x * W_;
  ApplyBiasAndActivation();
  return output_;
}

void RbmSpinV2::ApplyBiasAndActivation() {
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

}  // namespace netket
