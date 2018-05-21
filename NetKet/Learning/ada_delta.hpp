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

#ifndef NETKET_ADADELTA_HPP
#define NETKET_ADADELTA_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include "abstract_stepper.hpp"

namespace netket {

class AdaDelta {
  int npar_;

  // decay constant
  double rho_;

  // small parameter
  double eps_;

  Eigen::VectorXd Eg2_;
  Eigen::VectorXd Edx2_;

 public:
  AdaDelta(double rho = 0.95, double eps = 1.0e-6) : rho_(rho), eps_(eps) {
    npar_ = -1;
  }

  void SetNpar(int npar) {
    npar_ = npar;
    Eg2_.setZero(npar);
    Edx2_.setZero(npar);
  }

  void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) {
    assert(npar_ > 0);

    Eg2_ = rho_ * Eg2_ + (1. - rho_) * grad.cwiseAbs2();

    Eigen::VectorXd Dx(npar_);

    for (int i = 0; i < npar_; i++) {
      Dx(i) = -std::sqrt(Edx2_(i) + eps_) * grad(i);
      Dx(i) /= std::sqrt(Eg2_(i) + eps_);
      pars(i) += Dx(i);
    }

    Edx2_ = rho_ * Edx2_ + (1. - rho_) * Dx.cwiseAbs2();
  }

  void Reset() {
    Eg2_ = Eigen::VectorXd::Zero(npar_);
    Edx2_ = Eigen::VectorXd::Zero(npar_);
  }
};

}  // namespace netket

#endif
