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

#ifndef NETKET_STEPPER_HPP
#define NETKET_STEPPER_HPP

#include "abstract_stepper.hpp"
#include "ada_delta.hpp"
#include "ada_max.hpp"
#include "momentum.hpp"
#include "ams_grad.hpp"
#include "ada_grad.hpp"
#include "rms_prop.hpp"
#include "rprop.hpp"
#include "sgd.hpp"

namespace netket {

class Stepper : public AbstractStepper {
  using Ptype = std::unique_ptr<AbstractStepper>;

  Ptype s_;

 public:
  explicit Stepper(const json &pars) {
    if (!FieldExists(pars, "Learning")) {
      std::cerr << "Learning is not defined in the input" << std::endl;
      std::abort();
    }

    if (!FieldExists(pars["Learning"], "StepperType")) {
      std::cerr << "Stepper Type is not defined in the input" << std::endl;
      std::abort();
    }

    if (pars["Learning"]["StepperType"] == "Sgd") {
      s_ = Ptype(new Sgd(pars));
    } else if (pars["Learning"]["StepperType"] == "AdaMax") {
      s_ = Ptype(new AdaMax(pars));
    } else if (pars["Learning"]["StepperType"] == "AdaDelta") {
      s_ = Ptype(new AdaDelta(pars));
    } else if (pars["Learning"]["StepperType"] == "Momentum") {
      s_ = Ptype(new Momentum(pars));
    } else if (pars["Learning"]["StepperType"] == "AMSGrad") {
      s_ = Ptype(new AMSGrad(pars));
    } else if (pars["Learning"]["StepperType"] == "AdaGrad") {
      s_ = Ptype(new AdaGrad(pars));
    } else if (pars["Learning"]["StepperType"] == "RMSProp") {
      s_ = Ptype(new RMSProp(pars));
    } else {
      std::cout << "StepperType not found" << std::endl;
      std::abort();
    }
  }

  void Init(const Eigen::VectorXd &pars) override { return s_->Init(pars); }

  void Init(const Eigen::VectorXcd &pars) override { return s_->Init(pars); }

  void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) override {
    return s_->Update(grad, pars);
  }

  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXd &pars) override {
    return s_->Update(grad, pars);
  }

  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXcd &pars) override {
    return s_->Update(grad, pars);
  }

  void Reset() override { return s_->Reset(); }
};
}  // namespace netket
#endif
