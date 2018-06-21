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

#include "abstract_optimizer.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>

namespace netket {

class AdaGrad : public AbstractOptimizer {
  int npar_;

  double eta_;

  Eigen::VectorXd Gt_;

  double epscut_;

  const std::complex<double> I_;

public:
  // Json constructor
  explicit AdaGrad(const json &pars) : I_(0, 1) {
    npar_ = -1;

    from_json(pars);
    PrintParameters();
  }

  void PrintParameters() {
    InfoMessage() << "Adagrad optimizer initialized with these parameters :"
                  << std::endl;
    InfoMessage() << "Learning Rate = " << eta_ << std::endl;
    InfoMessage() << "Epscut = " << epscut_ << std::endl;
  }

  void Init(const Eigen::VectorXd &pars) override {
    npar_ = pars.size();
    Gt_.setZero(npar_);
  }

  void Init(const Eigen::VectorXcd &pars) override {
    npar_ = 2 * pars.size();
    Gt_.setZero(npar_);
  }

  void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) override {
    assert(npar_ > 0);

    Gt_ += grad.cwiseAbs2();

    for (int i = 0; i < npar_; i++) {
      pars(i) -= eta_ * grad(i) / std::sqrt(Gt_(i) + epscut_);
    }
  }

  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXd &pars) override {
    Update(Eigen::VectorXd(grad.real()), pars);
  }

  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXcd &pars) override {
    assert(npar_ == 2 * pars.size());

    for (int i = 0; i < pars.size(); i++) {
      Gt_(2 * i) += std::pow(grad(i).real(), 2);
      Gt_(2 * i + 1) += std::pow(grad(i).imag(), 2);
    }

    for (int i = 0; i < pars.size(); i++) {
      pars(i) -= eta_ * grad(i) / std::sqrt(Gt_(2 * i) + epscut_);
      pars(i) -= eta_ * I_ * grad(i) / std::sqrt(Gt_(2 * i + 1) + epscut_);
    }
  }

  void Reset() override { Gt_ = Eigen::VectorXd::Zero(npar_); }

  void from_json(const json &pars) {
    // DEPRECATED (to remove for v2.0.0)
    std::string section = "Optimizer";
    if (!FieldExists(pars, section)) {
      section = "Learning";
    }
    eta_ = FieldOrDefaultVal(pars[section], "LearningRate", 0.001);
    epscut_ = FieldOrDefaultVal(pars[section], "Epscut", 1.0e-7);
  }
};

} // namespace netket

#endif
