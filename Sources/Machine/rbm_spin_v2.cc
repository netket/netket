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

#include <pybind11/eigen.h>
#include <pybind11/eval.h>

#include "Utils/log_cosh.hpp"
#include "Utils/pybind_helpers.hpp"

namespace netket {

RbmSpinV2::RbmSpinV2(std::shared_ptr<const AbstractHilbert> hilbert,
                     Index nhidden, Index alpha, bool usea, bool useb,
                     Index const batch_size)
    : AbstractMachine{std::move(hilbert)},
      W_{},
      a_{nonstd::nullopt},
      b_{nonstd::nullopt},
      theta_{} {
  const auto nvisible = GetHilbert().Size();
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
  }
}

Eigen::VectorXcd RbmSpinV2::GetParameters() {
  Eigen::VectorXcd parameters(Npar());
  Index i = 0;
  if (a_.has_value()) {
    parameters.segment(i, a_->size()) = *a_;
    i += a_->size();
  }
  if (b_.has_value()) {
    parameters.segment(i, b_->size()) = *b_;
    i += b_->size();
  }
  parameters.segment(i, W_.size()) =
      Eigen::Map<Eigen::VectorXcd>(W_.data(), W_.size());
  return parameters;
}

void RbmSpinV2::SetParameters(Eigen::Ref<const Eigen::VectorXcd> parameters) {
  CheckShape(__FUNCTION__, "parameters", parameters.size(), Npar());
  Index i = 0;
  if (a_.has_value()) {
    *a_ = parameters.segment(i, a_->size());
    i += a_->size();
  }
  if (b_.has_value()) {
    *b_ = parameters.segment(i, b_->size());
    i += b_->size();
  }
  Eigen::Map<Eigen::VectorXcd>(W_.data(), W_.size()) =
      parameters.segment(i, W_.size());
}

void RbmSpinV2::LogVal(Eigen::Ref<const RowMatrix<double>> x,
                       Eigen::Ref<Eigen::VectorXcd> out, const any &) {
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", out.size(), x.rows());
  BatchSize(x.rows());
  if (a_.has_value()) {
    out.noalias() = x * (*a_);
  } else {
    out.setZero();
  }
  theta_.noalias() = x * W_;
  ApplyBiasAndActivation(out);
}

void RbmSpinV2::DerLog(Eigen::Ref<const RowMatrix<double>> x,
                       Eigen::Ref<RowMatrix<Complex>> out,
                       const any & /*unused*/) {
  CheckShape(__FUNCTION__, "v", {x.rows(), x.cols()},
             {std::ignore, Nvisible()});
  CheckShape(__FUNCTION__, "out", {out.rows(), out.cols()}, {x.rows(), Npar()});
  BatchSize(x.rows());

  auto i = Index{0};
  if (a_.has_value()) {
    out.block(0, i, BatchSize(), a_->size()) = x;
    i += a_->size();
  }

  Eigen::Map<RowMatrix<Complex>>{theta_.data(), theta_.rows(), theta_.cols()}
      .noalias() = x * W_;
  if (b_.has_value()) {
    theta_.array() = (theta_ + b_->transpose().colwise().replicate(BatchSize()))
                         .array()
                         .tanh();
    out.block(0, i, BatchSize(), b_->size()) = theta_;
    i += b_->size();
  } else {
    theta_.array() = theta_.array().tanh();
  }

  // TODO: Rewrite this using tensors
  omp_set_num_threads(2);
#pragma omp parallel for schedule(static)
  for (auto j = Index{0}; j < BatchSize(); ++j) {
    Eigen::Map<Eigen::MatrixXcd>{&out(j, i), W_.rows(), W_.cols()}.noalias() =
        x.row(j).transpose() * theta_.row(j);
  }
}

void RbmSpinV2::ApplyBiasAndActivation(Eigen::Ref<Eigen::VectorXcd> out) const {
  if (b_.has_value()) {
    omp_set_num_threads(2);
#pragma omp parallel for schedule(static)
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      out(j) += SumLogCoshBias(theta_.row(j), (*b_));  // total;
    }
  } else {
    omp_set_num_threads(2);
#pragma omp parallel for schedule(static)
    for (auto j = Index{0}; j < BatchSize(); ++j) {
      out(j) += SumLogCosh(theta_.row(j));
    }
  }
}

PyObject *RbmSpinV2::StateDict() {
  return ToStateDict(std::make_tuple(std::make_pair("a", std::ref(a_)),
                                     std::make_pair("b", std::ref(b_)),
                                     std::make_pair("w", std::ref(W_))));
}

}  // namespace netket
