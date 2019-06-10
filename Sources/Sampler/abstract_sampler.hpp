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

#ifndef NETKET_ABSTRACTSAMPLER_HPP
#define NETKET_ABSTRACTSAMPLER_HPP

#include <functional>
#include <memory>
#include <vector>
#include "Hilbert/abstract_hilbert.hpp"

namespace netket {

class AbstractSampler {
 public:
  using MachineFunction = std::function<double(const Complex&)>;

  virtual void Reset(bool initrandom = false) = 0;

  virtual void Sweep() = 0;

  virtual Eigen::VectorXd Visible() = 0;

  virtual void SetVisible(const Eigen::VectorXd& v) = 0;

  virtual Eigen::VectorXd Acceptance() const = 0;

  // Computes the derivative of the machine on the current visible
  // Using the lookUp tables if possible
  virtual AbstractMachine::VectorType DerLogVisible() {
    return GetMachine().DerLog(Visible());
  }

  virtual ~AbstractSampler() {}

  void Seed(DistributedRandomEngine::ResultType base_seed) {
    engine_.Seed(base_seed);
    this->Reset(true);
  }

  virtual void SetMachineFunc(MachineFunction machine_func) {
    if (!machine_func) {
      throw RuntimeError{"Invalid machine function in Sampler"};
    }

    machine_func_ = std::move(machine_func);
  }

  std::shared_ptr<const AbstractHilbert> GetHilbertShared() const noexcept {
    return hilbert_;
  }

  const AbstractHilbert &GetHilbert() const noexcept {
    return *hilbert_;
  }

  AbstractMachine &GetMachine() const noexcept {
    return psi_;
  }

  const MachineFunction& GetMachineFunc() const noexcept {
    return machine_func_;
  }

 protected:
  AbstractSampler(AbstractMachine& psi)
      : psi_(psi), hilbert_(psi.GetHilbertShared()) {
    // Default initialization for the machine function to be sampled from
    machine_func_ = static_cast<double (*)(const Complex&)>(&std::norm);
  }

  default_random_engine& GetRandomEngine() { return engine_.Get(); }

 private:
  DistributedRandomEngine engine_;
  MachineFunction machine_func_;
  AbstractMachine& psi_;
  std::shared_ptr<const AbstractHilbert> hilbert_;
};

}  // namespace netket
#endif
