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

#ifndef NETKET_ADAMAX_HPP
#define NETKET_ADAMAX_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include "abstract_optimizer.hpp"

namespace netket {

class AdaMax : public AbstractOptimizer {
  int npar_;

  double alpha_;
  double beta1_;
  double beta2_;

  Eigen::VectorXd ut_;
  Eigen::VectorXd mt_;

  double niter_;
  double niter_reset_;

  double epscut_;

 public:
  explicit AdaMax(double alpha = 0.001, double beta1 = 0.9,
                  double beta2 = 0.999, double epscut = 1.0e-7)
      : alpha_(alpha), beta1_(beta1), beta2_(beta2), epscut_(epscut) {
    npar_ = -1;
    niter_ = 0;
    niter_reset_ = -1;

    PrintParameters();
  }

  void PrintParameters() {
    InfoMessage() << "Adamax optimizer initialized with these parameters :"
                  << std::endl;
    InfoMessage() << "Alpha = " << alpha_ << std::endl;
    InfoMessage() << "Beta1 = " << beta1_ << std::endl;
    InfoMessage() << "Beta2 = " << beta2_ << std::endl;
    InfoMessage() << "Epscut = " << epscut_ << std::endl;
  }

  void Init(int npar) override {
    npar_ = npar;
    ut_.setZero(npar_);
    mt_.setZero(npar_);

    niter_ = 0;
  }

  void Update(const Eigen::VectorXd &grad,
              Eigen::Ref<Eigen::VectorXd> pars) override {
    assert(npar_ > 0);

    mt_ = beta1_ * mt_ + (1. - beta1_) * grad;

    for (int i = 0; i < npar_; i++) {
      ut_(i) = std::max(std::max(std::abs(grad(i)), beta2_ * ut_(i)), epscut_);
    }
    niter_ += 1.;
    if (niter_reset_ > 0) {
      if (niter_ > niter_reset_) {
        niter_ = 1;
      }
    }

    double eta = alpha_ / (1. - std::pow(beta1_, niter_));
    for (int i = 0; i < npar_; i++) {
      pars(i) -= eta * mt_(i) / ut_(i);
    }
  }

  void Reset() override {
    ut_ = Eigen::VectorXd::Zero(npar_);
    mt_ = Eigen::VectorXd::Zero(npar_);
    niter_ = 0;
  }

  void SetResetEvery(double niter_reset) { niter_reset_ = niter_reset; }
};

}  // namespace netket

#endif
