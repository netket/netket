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

#ifndef NETKET_RPROP_HH
#define NETKET_RPROP_HH

#include "abstract_stepper.hh"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>

namespace netket {

class Rprop {

  // decay constant
  double etap_;
  double etam_;

  int npar_;

  Eigen::VectorXd oldgrad_;
  Eigen::VectorXd delta_;

  double deltamin_;
  double deltamax_;
  double delta0_;

public:
  Rprop(double etam, double etap, double delta0, double deltamin,
        double deltamax)
      : etap_(etap), etam_(etam), deltamin_(deltamin), deltamax_(deltamax),
        delta0_(delta0) {
    npar_ = -1;
  }

  void SetNpar(int npar) {
    npar_ = npar;

    oldgrad_.resize(npar_);
    oldgrad_ = Eigen::VectorXd::Ones(npar_);
    delta_.resize(npar_);
    delta_ = delta0_ * Eigen::VectorXd::Ones(npar_);
  }

  void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) {
    assert(npar_ > 0);
    double normgrad = 1. / std::sqrt(grad.norm());

    for (int i = 0; i < npar_; i++) {
      if (grad(i) * oldgrad_(i) > 0) {
        delta_(i) = std::min(deltamax_, delta_(i) * etap_);
        oldgrad_(i) = grad(i);
      } else if (grad(i) * oldgrad_(i) < 0) {
        delta_(i) = std::max(deltamin_, delta_(i) * etam_);
        oldgrad_(i) = 0;
      } else {
        oldgrad_(i) = grad(i);
      }
      if (grad(i) > 0) {
        pars(i) -= std::min(delta_(i), normgrad);
      } else if (grad(i) < 0) {
        pars(i) += std::min(delta_(i), normgrad);
      }
    }
    std::cerr << delta_.mean() << std::endl;
  }

  void Reset() {}
};

} // namespace netket

#endif
