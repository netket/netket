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
#include <complex>
#include <iostream>
#include "abstract_optimizer.hpp"

namespace netket {

class AdaDelta : public AbstractOptimizer {
  int npar_;

  double rho_;
  double epscut_;

  Eigen::VectorXd Eg2_;
  Eigen::VectorXd Edx2_;

  const Complex I_;

 public:
  // Json constructor
  explicit AdaDelta(double rho = 0.95, double epscut = 1.0e-7)
      : rho_(rho), epscut_(epscut), I_(0, 1) {
    npar_ = -1;

    PrintParameters();
  }

  void PrintParameters() {
    InfoMessage() << "Adadelta optimizer initialized with these parameters :"
                  << std::endl;
    InfoMessage() << "Rho = " << rho_ << std::endl;
    InfoMessage() << "Epscut = " << epscut_ << std::endl;
  }

  void Init(int npar) override {
    npar_ = npar;
    Eg2_.setZero(npar_);
    Edx2_.setZero(npar_);
  }

  void Update(const Eigen::VectorXd &grad,
              Eigen::Ref<Eigen::VectorXd> pars) override {
    assert(npar_ > 0);

    Eg2_ = rho_ * Eg2_ + (1. - rho_) * grad.cwiseAbs2();

    Eigen::VectorXd Dx(npar_);

    for (int i = 0; i < npar_; i++) {
      Dx(i) = -std::sqrt(Edx2_(i) + epscut_) * grad(i);
      Dx(i) /= std::sqrt(Eg2_(i) + epscut_);
      pars(i) += Dx(i);
    }

    Edx2_ = rho_ * Edx2_ + (1. - rho_) * Dx.cwiseAbs2();
  }

  void Reset() override {
    Eg2_ = Eigen::VectorXd::Zero(npar_);
    Edx2_ = Eigen::VectorXd::Zero(npar_);
  }
};

}  // namespace netket

#endif
