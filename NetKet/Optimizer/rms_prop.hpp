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

#ifndef NETKET_RMSPROP_HPP
#define NETKET_RMSPROP_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include "abstract_optimizer.hpp"

namespace netket {

class RMSProp : public AbstractOptimizer {
  int npar_;

  double eta_;
  double beta_;

  Eigen::VectorXd st_;

  double epscut_;

 public:
  explicit RMSProp(double eta = 0.001, double beta = 0.9,
                   double epscut = 1.0e-7)
      : eta_(eta), beta_(beta), epscut_(epscut) {
    npar_ = -1;
    PrintParameters();
  }

  void PrintParameters() {
    InfoMessage() << "RMSProp optimizer initialized with these parameters :"
                  << std::endl;
    InfoMessage() << "Learning Rate = " << eta_ << std::endl;
    InfoMessage() << "Beta = " << beta_ << std::endl;
    InfoMessage() << "Epscut = " << epscut_ << std::endl;
  }

  void Init(int npar) override {
    npar_ = npar;
    st_.setZero(npar_);
  }

  void Update(const Eigen::VectorXd &grad,
              Eigen::Ref<Eigen::VectorXd> pars) override {
    assert(npar_ > 0);

    st_ = beta_ * st_ + (1. - beta_) * grad.cwiseAbs2();

    for (int i = 0; i < npar_; i++) {
      pars(i) -= eta_ * grad(i) / (std::sqrt(st_(i)) + epscut_);
    }
  }

  void Reset() override { st_ = Eigen::VectorXd::Zero(npar_); }
};

}  // namespace netket

#endif
