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

#ifndef NETKET_OPTIMIZER_HPP
#define NETKET_OPTIMIZER_HPP

#include "Utils/all_utils.hpp"
#include "abstract_optimizer.hpp"
#include "ada_delta.hpp"
#include "ada_grad.hpp"
#include "ada_max.hpp"
#include "ams_grad.hpp"
#include "momentum.hpp"
#include "mpark/variant.hpp"
#include "rms_prop.hpp"
#include "sgd.hpp"

namespace netket {

class Optimizer : public AbstractOptimizer {
  using VariantType = mpark::variant<AdaDelta, AdaGrad, AdaMax, AMSGrad,
                                     Momentum, RMSProp, Sgd>;

  VariantType obj_;

 public:
  explicit Optimizer(VariantType obj) : obj_(obj) {}

  void Init(const Eigen::VectorXd &pars) override {
    mpark::visit([&pars](auto &&obj) { obj.Init(pars); }, obj_);
  }
  void Init(const Eigen::VectorXcd &pars) override {
    mpark::visit([&pars](auto &&obj) { obj.Init(pars); }, obj_);
  }
  void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) override {
    mpark::visit([&grad, &pars](auto &&obj) { obj.Update(grad, pars); }, obj_);
  }
  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXd &pars) override {
    mpark::visit([&grad, &pars](auto &&obj) { obj.Update(grad, pars); }, obj_);
  }
  void Update(const Eigen::VectorXcd &grad, Eigen::VectorXcd &pars) override {
    mpark::visit([&grad, &pars](auto &&obj) { obj.Update(grad, pars); }, obj_);
  }
  void Reset() override {
    mpark::visit([](auto &&obj) { obj.Reset(); }, obj_);
  }
};
//
// class Optimizer : public AbstractOptimizer {
//   using Ptype = std::unique_ptr<AbstractOptimizer>;
//
//   Ptype s_;
//
//   bool usenormclipping_;
//   bool usevalclipping_;
//
//   double clipnorm_;
//   double clipval_;
//
//  public:
//   explicit Optimizer(const json &pars) {
//     std::string optimizer_name = "none specified";
//
//     if (FieldExists(pars, "Optimizer")) {
//       optimizer_name = FieldVal(pars["Optimizer"], "Name", "Optimizer");
//     } else if (FieldExists(pars, "Learning")) {
//       // DEPRECATED (to remove for v2.0.0)
//       optimizer_name =
//           FieldVal(pars["Learning"], "StepperType", "Learning/Optimizer");
//       WarningMessage()
//           << "Declaring Optimizers within the Learning section is "
//              "deprecated.\n Please use the dedicated Optimizer section.\n";
//     }
//
//     if (optimizer_name == "Sgd") {
//       s_ = Ptype(new Sgd(pars));
//     } else if (optimizer_name == "AdaMax") {
//       s_ = Ptype(new AdaMax(pars));
//     } else if (optimizer_name == "AdaDelta") {
//       s_ = Ptype(new AdaDelta(pars));
//     } else if (optimizer_name == "Momentum") {
//       s_ = Ptype(new Momentum(pars));
//     } else if (optimizer_name == "AMSGrad") {
//       s_ = Ptype(new AMSGrad(pars));
//     } else if (optimizer_name == "AdaGrad") {
//       s_ = Ptype(new AdaGrad(pars));
//     } else if (optimizer_name == "RMSProp") {
//       s_ = Ptype(new RMSProp(pars));
//     } else {
//       std::stringstream s;
//       s << "Unknown Optimizer Name: " << optimizer_name;
//       throw InvalidInputError(s.str());
//     }
//
//     if (FieldExists(pars["Optimizer"], "ClipNorm")) {
//       usenormclipping_ = true;
//       clipnorm_ = FieldVal(pars["Optimizer"], "ClipNorm");
//       InfoMessage() << "Clipping by Norm to " << clipnorm_ << std::endl;
//     } else {
//       usenormclipping_ = false;
//       clipnorm_ = 0.0;
//     }
//
//     if (FieldExists(pars["Optimizer"], "ClipVal")) {
//       usenormclipping_ = true;
//       clipval_ = FieldVal(pars["Optimizer"], "ClipVal");
//       InfoMessage() << "Clipping by Value to " << clipval_ << std::endl;
//     } else {
//       usevalclipping_ = false;
//       clipval_ = 0.0;
//     }
//   }
//
//   void Init(const Eigen::VectorXd &pars) override { return s_->Init(pars); }
//
//   void Init(const Eigen::VectorXcd &pars) override { return s_->Init(pars); }
//
//   void Update(const Eigen::VectorXd &grad, Eigen::VectorXd &pars) override {
//     if (usenormclipping_ || usevalclipping_) {
//       Eigen::VectorXd clippedgrad = grad;
//       if (usenormclipping_) {
//         ClipNorm<Eigen::VectorXd>(clippedgrad);
//       }
//       if (usevalclipping_) {
//         ClipVal<Eigen::VectorXd>(clippedgrad);
//       }
//       return s_->Update(clippedgrad, pars);
//     } else {
//       return s_->Update(grad, pars);
//     }
//   }
//
//   void Update(const Eigen::VectorXcd &grad, Eigen::VectorXd &pars) override {
//     if (usenormclipping_ || usevalclipping_) {
//       Eigen::VectorXcd clippedgrad = grad;
//       if (usenormclipping_) {
//         ClipNorm<Eigen::VectorXcd>(clippedgrad);
//       }
//       if (usevalclipping_) {
//         ClipVal<Eigen::VectorXcd>(clippedgrad);
//       }
//       return s_->Update(clippedgrad, pars);
//     } else {
//       return s_->Update(grad, pars);
//     }
//   }
//
//   void Update(const Eigen::VectorXcd &grad, Eigen::VectorXcd &pars) override
//   {
//     if (usenormclipping_ || usevalclipping_) {
//       Eigen::VectorXcd clippedgrad = grad;
//       if (usenormclipping_) {
//         ClipNorm<Eigen::VectorXcd>(clippedgrad);
//       }
//       if (usevalclipping_) {
//         ClipVal<Eigen::VectorXcd>(clippedgrad);
//       }
//       return s_->Update(clippedgrad, pars);
//     } else {
//       return s_->Update(grad, pars);
//     }
//   }
//
//   template <typename T>
//   inline void ClipNorm(T &grad) {
//     double norm = grad.norm();
//     if (norm > clipnorm_) {
//       grad *= clipnorm_ / norm;
//     }
//   }
//
//   template <typename T>
//   inline void ClipVal(T &grad) {
//     int length = grad.size();
//     for (int i override {} i < length; ++i) {
//       double val = std::abs(grad(i));
//       if (val > clipval_) {
//         grad(i) *= clipval_ / val;
//       }
//     }
//   }
//
//   void Reset() override { return s_->Reset(); }
// };
}  // namespace netket
#endif
