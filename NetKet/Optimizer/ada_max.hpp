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

#include "abstract_optimizer.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>

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

  const std::complex<double> I_;

public:
  // Json constructor
  explicit AdaMax(const json &pars) : I_(0, 1) {
    npar_ = -1;
    niter_ = 0;
    niter_reset_ = -1;

    from_json(pars);
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

  void Init(const Eigen::VectorXd &pars) override {
    npar_ = pars.size();
    ut_.setZero(npar_);
    mt_.setZero(npar_);

    niter_ = 0;
  }

  void Init(const Eigen::VectorXcd &pars) override {
    npar_ = 2 * pars.size();
    ut_.setZero(npar_);
    mt_.setZero(npar_);

    niter_ = 0;
  }

  void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) override {
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
      ut_(2 * i) = std::max(
          std::max(std::abs(grad(i).real()), beta2_ * ut_(2 * i)), epscut_);
      ut_(2 * i + 1) = std::max(
          std::max(std::abs(grad(i).imag()), beta2_ * ut_(2 * i + 1)), epscut_);
    }

    niter_ += 1.;
    if (niter_reset_ > 0) {
      if (niter_ > niter_reset_) {
        niter_ = 1;
      }
    }

    double eta = alpha_ / (1. - std::pow(beta1_, niter_));
    for (int i = 0; i < pars.size(); i++) {
      pars(i) -= eta * mt_(2 * i) / ut_(2 * i);
      pars(i) -= eta * I_ * mt_(2 * i + 1) / ut_(2 * i + 1);
    }
  }

  void Reset() override {
    ut_ = Eigen::VectorXd::Zero(npar_);
    mt_ = Eigen::VectorXd::Zero(npar_);
    niter_ = 0;
  }

  void SetResetEvery(double niter_reset) { niter_reset_ = niter_reset; }

  void from_json(const json &pars) {
    // DEPRECATED (to remove for v2.0.0)
    std::string section = "Optimizer";
    if (!FieldExists(pars, section)) {
      section = "Learning";
    }

    alpha_ = FieldOrDefaultVal(pars[section], "Alpha", 0.001);
    beta1_ = FieldOrDefaultVal(pars[section], "Beta1", 0.9);
    beta2_ = FieldOrDefaultVal(pars[section], "Beta2", 0.999);
    epscut_ = FieldOrDefaultVal(pars[section], "Epscut", 1.0e-7);
  }
};

} // namespace netket

#endif
