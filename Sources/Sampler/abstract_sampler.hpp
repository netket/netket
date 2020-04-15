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
#include "Machine/abstract_machine.hpp"

namespace netket {

class AbstractSampler {
 public:
  virtual void Reset(bool initrandom = false) = 0;

  virtual void Sweep() = 0;

  virtual std::pair<Eigen::Ref<const RowMatrix<double>>,
                    Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const = 0;

  virtual void SetVisible(Eigen::Ref<const RowMatrix<double>> v) = 0;

  virtual ~AbstractSampler() {}

  void Seed(DistributedRandomEngine::ResultType base_seed) {
    GetDistributedRandomEngine().Seed(base_seed);
    this->Reset(true);
  }

  virtual void SetMachinePow(double machine_pow) { machine_pow_ = machine_pow; }

  double GetMachinePow() const noexcept { return machine_pow_; }

  AbstractMachine& GetMachine() const noexcept { return psi_; }

  virtual Index BatchSize() const noexcept = 0;

  virtual Index NChains() const noexcept = 0;

 protected:
  AbstractSampler(AbstractMachine& psi) : machine_pow_{2.0}, psi_{psi} {}

 private:
  double machine_pow_;
  AbstractMachine& psi_;
};  // namespace netket

inline Eigen::Ref<const Eigen::VectorXd> VisibleLegacy(
    const AbstractSampler& sampler) {
  auto state = sampler.CurrentState();
  const auto& visible = state.first;
  const auto& log_val = state.second;
  NETKET_CHECK(visible.cols() == sampler.GetMachine().Nvisible(),
               std::runtime_error,
               "bug in CurrentState(): wrong number of columns: "
                   << visible.cols() << "; expected "
                   << sampler.GetMachine().Nvisible());
  NETKET_CHECK(visible.rows() == 1, std::runtime_error,
               "bug in CurrentState(): `visible` has wrong number of rows: "
                   << visible.rows() << "; expected 1");
  NETKET_CHECK(log_val.size() == 1, std::runtime_error,
               "bug in CurrentState(): `log_val` has wrong size: "
                   << log_val.size() << "; expected 1");
  return visible.row(0);
}

namespace detail {
inline Index CheckNChains(const char* func, const Index n_chains) {
  if (n_chains <= 0) {
    std::ostringstream msg;
    msg << func << ": invalid number of chains: " << n_chains
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  return n_chains;
}

inline Index CheckSweepSize(const char* func, const Index sweep_size) {
  if (sweep_size <= 0) {
    std::ostringstream msg;
    msg << func << ": invalid sweep size: " << sweep_size
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  return sweep_size;
}

}  // namespace detail

#define NETKET_SAMPLER_ACCEPTANCE_DEFAULT(accepts, moves)                  \
  double Acceptance() const {                                              \
    NETKET_CHECK(moves > 0, RuntimeError,                                  \
                 "Cannot compute acceptance, because no moves were made"); \
    return static_cast<double>(accepts) / static_cast<double>(moves);      \
  }

}  // namespace netket
#endif
