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
#include "abstract_stepper.hpp"

namespace netket {

class AdaMax : public AbstractStepper {
  int npar_;

  double alpha_;
  double beta1_;
  double beta2_;

  Eigen::VectorXd ut_;
  Eigen::VectorXd mt_;

  double niter_;
  double niter_reset_;

  double epscut_;

  int mynode_;

  const std::complex<double> I_;

 public:
  // Json constructor
  explicit AdaMax(const json &pars)
      : alpha_(FieldOrDefaultVal(pars["Learning"], "Alpha", 0.001)),
        beta1_(FieldOrDefaultVal(pars["Learning"], "Beta1", 0.9)),
        beta2_(FieldOrDefaultVal(pars["Learning"], "Beta2", 0.999)),
        epscut_(FieldOrDefaultVal(pars["Learning"], "Epscut", 1.0e-7)),
        I_(0, 1) {
    npar_ = -1;
    niter_ = 0;
    niter_reset_ = -1;

    PrintParameters();
  }

  void PrintParameters() {
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
    if (mynode_ == 0) {
      std::cout << "# Adamax stepper initialized with these parameters : "
                << std::endl;
      std::cout << "# Alpha = " << alpha_ << std::endl;
      std::cout << "# Beta1 = " << beta1_ << std::endl;
      std::cout << "# Beta2 = " << beta2_ << std::endl;
      std::cout << "# Epscut = " << epscut_ << std::endl;
    }
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
};

}  // namespace netket

#endif
