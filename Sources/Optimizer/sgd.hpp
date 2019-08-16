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

#ifndef NETKET_SGD_HPP
#define NETKET_SGD_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include "abstract_optimizer.hpp"

namespace netket {

class Sgd : public AbstractOptimizer {
  // decay constant
  double eta_;

  int npar_;

  double l2reg_;

  double decay_factor_;

 public:
  explicit Sgd(double learning_rate, double l2reg = 0,
               double decay_factor = 1.0)
      : eta_(learning_rate), l2reg_(l2reg) {
    npar_ = -1;
    SetDecayFactor(decay_factor);
    PrintParameters();
  }

  void PrintParameters() {
    InfoMessage() << "Sgd optimizer initialized with these parameters :"
                  << std::endl;
    InfoMessage() << "Learning Rate = " << eta_ << std::endl;
    InfoMessage() << "L2 Regularization = " << l2reg_ << std::endl;
    InfoMessage() << "Decay Factor = " << decay_factor_ << std::endl;
  }

  void Init(int npar) override { npar_ = npar; }

  void Update(const Eigen::VectorXd &grad,
              Eigen::Ref<Eigen::VectorXd> pars) override {
    assert(npar_ > 0);

    eta_ *= decay_factor_;

    for (int i = 0; i < npar_; i++) {
      pars(i) = pars(i) - (grad(i) + l2reg_ * pars(i)) * eta_;
    }
  }

  void SetDecayFactor(double decay_factor) {
    assert(decay_factor <= 1.00001);
    decay_factor_ = decay_factor;
  }

  void Reset() override {}
};

}  // namespace netket

#endif
