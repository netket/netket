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

#ifndef NETKET_ABSTRACT_OPTIMIZER_HPP
#define NETKET_ABSTRACT_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <complex>
#include <vector>

namespace netket {

class AbstractOptimizer {
 public:
  virtual void Init(int npar) = 0;

  virtual void Update(const Eigen::VectorXd &grad,
                      Eigen::Ref<Eigen::VectorXd> pars) = 0;

  virtual void Reset() = 0;

  void Init(int npar, bool is_holomorphic) {
    is_holomorphic_ = is_holomorphic;

    if (is_holomorphic) {
      Init(2 * npar);
    } else {
      Init(npar);
    }
  }

  virtual void Update(const Eigen::VectorXcd &grad,
                      Eigen::Ref<Eigen::VectorXcd> pars) {
    auto npar = pars.size();

    if (is_holomorphic_) {
      Eigen::VectorXd gradr(2 * npar);
      gradr << grad.real(), grad.imag();
      Eigen::VectorXd parsr(2 * npar);
      parsr << pars.real(), pars.imag();
      Update(gradr, parsr);
      pars.real() = parsr.head(npar);
      pars.imag() = parsr.tail(npar);
    } else {
      Eigen::VectorXd pp(pars.real());
      Update(grad.real(), pp);
      pars.real() = pp;
    }
  }

  virtual ~AbstractOptimizer() {}

 private:
  bool is_holomorphic_;
};
}  // namespace netket

#endif
