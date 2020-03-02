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

#ifndef NETKET_RBM_SPIN_KERNEL_HPP
#define NETKET_RBM_SPIN_KERNEL_HPP

#include "Utils/log_cosh.hpp"

namespace netket {

struct RbmSpinKernel {
  RowMatrix<Complex> theta_;

  void LogVal(Eigen::Ref<const RowMatrix<double>> x,
              Eigen::Ref<Eigen::VectorXcd> out,
              Eigen::Ref<const Eigen::MatrixXcd> W,
              nonstd::optional<Eigen::Ref<const VectorXcd>> a,
              nonstd::optional<Eigen::Ref<const VectorXcd>> b) {
    CheckShape(__FUNCTION__, "out", out.size(), x.rows());

    if (a.has_value()) {
      out.noalias() = x * (*a);
    } else {
      out.setZero();
    }

    if (x.rows() != theta_.rows() || theta_.cols() != W.cols()) {
      theta_.resize(x.rows(), W.cols());
    }

    theta_.noalias() = x * W.transpose();

    if (b.has_value()) {
#pragma omp parallel for schedule(static)
      for (auto j = Index{0}; j < x.rows(); ++j) {
        out(j) += SumLogCoshBias(theta_.row(j), (*b));  // total;
      }
    } else {
#pragma omp parallel for schedule(static)
      for (auto j = Index{0}; j < x.rows(); ++j) {
        out(j) += SumLogCosh(theta_.row(j));
      }
    }
  }
};

}  // namespace netket

#endif
