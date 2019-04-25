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

#ifndef NETKET_ADAGRAD_HPP
#define NETKET_ADAGRAD_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include "abstract_optimizer.hpp"

namespace netket {

class AdaGrad : public AbstractOptimizer {
  int npar_;

  double eta_;

  Eigen::VectorXd Gt_;

  double epscut_;

 public:
  explicit AdaGrad(double eta = 0.001, double epscut = 1.0e-7)
      : eta_(eta), epscut_(epscut) {
    npar_ = -1;

    PrintParameters();
  }

  void PrintParameters() {
    InfoMessage() << "Adagrad optimizer initialized with these parameters :"
                  << std::endl;
    InfoMessage() << "Learning Rate = " << eta_ << std::endl;
    InfoMessage() << "Epscut = " << epscut_ << std::endl;
  }

  void Init(int npar) override {
    npar_ = npar;
    Gt_.setZero(npar_);
  }

  void Update(const Eigen::VectorXd &grad,
              Eigen::Ref<Eigen::VectorXd> pars) override {
    assert(npar_ > 0);

    Gt_ += grad.cwiseAbs2();

    for (int i = 0; i < npar_; i++) {
      pars(i) -= eta_ * grad(i) / std::sqrt(Gt_(i) + epscut_);
    }
  }

  void Reset() override { Gt_ = Eigen::VectorXd::Zero(npar_); }
};

}  // namespace netket

#endif
