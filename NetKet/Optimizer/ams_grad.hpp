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

#ifndef NETKET_AMSGRAD_HPP
#define NETKET_AMSGRAD_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include "abstract_optimizer.hpp"

namespace netket {

class AMSGrad : public AbstractOptimizer {
  int npar_;

  double eta_;
  double beta1_;
  double beta2_;

  Eigen::VectorXd mt_;
  Eigen::VectorXd vt_;

  double epscut_;

 public:
  explicit AMSGrad(double eta = 0.001, double beta1 = 0.9, double beta2 = 0.999,
                   double epscut = 1.0e-7)
      : eta_(eta), beta1_(beta1), beta2_(beta2), epscut_(epscut) {
    npar_ = -1;

    PrintParameters();
  }

  void PrintParameters() {
    InfoMessage() << "AMSGrad optimizer initialized with these parameters :"
                  << std::endl;
    InfoMessage() << "Learning Rate = " << eta_ << std::endl;
    InfoMessage() << "Beta1 = " << beta1_ << std::endl;
    InfoMessage() << "Beta2 = " << beta2_ << std::endl;
    InfoMessage() << "Epscut = " << epscut_ << std::endl;
  }

  void Init(int npar) override {
    npar_ = npar;
    mt_.setZero(npar_);
    vt_.setZero(npar_);
  }

  void Update(const Eigen::VectorXd &grad,
              Eigen::Ref<Eigen::VectorXd> pars) override {
    assert(npar_ > 0);

    mt_ = beta1_ * mt_ + (1. - beta1_) * grad;

    for (int i = 0; i < npar_; i++) {
      vt_(i) = std::max(vt_(i),
                        beta2_ * vt_(i) + (1 - beta2_) * std::pow(grad(i), 2));
    }

    for (int i = 0; i < npar_; i++) {
      pars(i) -= eta_ * mt_(i) / (std::sqrt(vt_(i)) + epscut_);
    }
  }

  void Reset() override {
    mt_ = Eigen::VectorXd::Zero(npar_);
    vt_ = Eigen::VectorXd::Zero(npar_);
  }
};

}  // namespace netket

#endif
