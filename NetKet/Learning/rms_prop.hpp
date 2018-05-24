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
#include "abstract_stepper.hpp"

namespace netket {

class RMSProp : public AbstractStepper {
  int npar_;

  double eta_;
  double beta_;

  Eigen::VectorXd st_;

  double epscut_;

  int mynode_;

  const std::complex<double> I_;

 public:
  // Json constructor
  explicit RMSProp(const json &pars)
      : eta_(FieldOrDefaultVal(pars["Learning"],"LearningRate",0.001)),
        beta_(FieldOrDefaultVal(pars["Learning"],"Beta",0.9)),
        epscut_(FieldOrDefaultVal(pars["Learning"],"Epscut",1.0e-8)),
        I_(0, 1) {
    npar_ = -1;

    PrintParameters();
  }

  void PrintParameters() {
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode_);
    if (mynode_ == 0) {
      std::cout << "# RMSProp stepper initialized with these parameters : "
                << std::endl;
      std::cout << "# Learning Rate = " << eta_ << std::endl;
      std::cout << "# Beta = " << beta_ << std::endl;
      std::cout << "# Epscut = " << epscut_ << std::endl;
    }
  }

  void Init(const Eigen::VectorXd &pars) override {
    npar_ = pars.size();
    st_.setZero(npar_);
  }

  void Init(const Eigen::VectorXcd &pars) override {
    npar_ = 2 * pars.size();
    st_.setZero(npar_);
  }

  void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) override {
    assert(npar_ > 0);

    st_=beta_*st_+(1.-beta_)*grad.cwiseAbs2();

    for(int i=0;i<npar_;i++){
      pars(i)-=eta_*grad(i)/(std::sqrt(st_(i))+epscut_);
    }
  }

  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXd &pars) override {
    Update(Eigen::VectorXd(grad.real()), pars);
  }

  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXcd &pars) override {
    assert(npar_ == 2 * pars.size());

    for(int i=0;i<pars.size();i++){
      st_(2*i)=beta_*st_(2*i)+(1.-beta_)*std::pow(grad(i).real(),2);
      st_(2*i+1)=beta_*st_(2*i+1)+(1.-beta_)*std::pow(grad(i).imag(),2);
      pars(i)-=eta_*grad(i).real()/(std::sqrt(st_(2*i))+epscut_);
      pars(i)-=eta_*I_*grad(i).imag()/(std::sqrt(st_(2*i+1))+epscut_);
    }
  }

  void Reset() override {
    st_ = Eigen::VectorXd::Zero(npar_);
  }

};

}  // namespace netket

#endif
