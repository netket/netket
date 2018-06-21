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

#include "abstract_optimizer.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>

namespace netket {

class AMSGrad : public AbstractOptimizer {
  int npar_;

  double eta_;
  double beta1_;
  double beta2_;

  Eigen::VectorXd mt_;
  Eigen::VectorXd vt_;

  double epscut_;

  const std::complex<double> I_;

public:
  // Json constructor
  explicit AMSGrad(const json &pars) : I_(0, 1) {
    npar_ = -1;
    from_json(pars);
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

  void Init(const Eigen::VectorXd &pars) override {
    npar_ = pars.size();
    mt_.setZero(npar_);
    vt_.setZero(npar_);
  }

  void Init(const Eigen::VectorXcd &pars) override {
    npar_ = 2 * pars.size();
    mt_.setZero(npar_);
    vt_.setZero(npar_);
  }

  void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) override {
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

  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXd &pars) override {
    Update(Eigen::VectorXd(grad.real()), pars);
  }

  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXcd &pars) override {
    assert(npar_ == 2 * pars.size());

    for (int i = 0; i < pars.size(); i++) {
      mt_(2 * i) = beta1_ * mt_(2 * i) + (1. - beta1_) * grad(i).real();
      mt_(2 * i + 1) = beta1_ * mt_(2 * i + 1) + (1. - beta1_) * grad(i).imag();
    }

    for (int i = 0; i < pars.size(); i++) {
      vt_(2 * i) =
          std::max(vt_(2 * i), beta2_ * vt_(2 * i) +
                                   (1 - beta2_) * std::pow(grad(i).real(), 2));
      vt_(2 * i + 1) = std::max(vt_(2 * i + 1),
                                beta2_ * vt_(2 * i + 1) +
                                    (1 - beta2_) * std::pow(grad(i).imag(), 2));
    }

    for (int i = 0; i < pars.size(); i++) {
      pars(i) -= eta_ * mt_(2 * i) / (std::sqrt(vt_(2 * i)) + epscut_);
      pars(i) -=
          eta_ * I_ * mt_(2 * i + 1) / (std::sqrt(vt_(2 * i + 1)) + epscut_);
    }
  }

  void Reset() override {
    mt_ = Eigen::VectorXd::Zero(npar_);
    vt_ = Eigen::VectorXd::Zero(npar_);
  }

  void from_json(const json &pars) {
    // DEPRECATED (to remove for v2.0.0)
    std::string section = "Optimizer";
    if (!FieldExists(pars, section)) {
      section = "Learning";
    }
    eta_ = FieldOrDefaultVal(pars[section], "LearningRate", 0.001);
    beta1_ = FieldOrDefaultVal(pars[section], "Beta1", 0.9);
    beta2_ = FieldOrDefaultVal(pars[section], "Beta2", 0.999);
    epscut_ = FieldOrDefaultVal(pars[section], "Epscut", 1.0e-7);
  }
};

} // namespace netket

#endif
