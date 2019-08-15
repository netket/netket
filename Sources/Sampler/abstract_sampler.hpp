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
  using MachineFunction =
      std::function<void(nonstd::span<const Complex>, nonstd::span<double>)>;

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

  virtual void SetMachineFunc(MachineFunction machine_func) {
    NETKET_CHECK(machine_func, InvalidInputError,
                 "Invalid machine function in Sampler");
    machine_func_ = std::move(machine_func);
  }

  AbstractMachine& GetMachine() const noexcept { return psi_; }

  const MachineFunction& GetMachineFunc() const noexcept {
    return machine_func_;
  }

  virtual Index BatchSize() const noexcept = 0;

 protected:
  AbstractSampler(AbstractMachine& psi)
      : machine_func_{[](nonstd::span<const Complex> x,
                         nonstd::span<double> out) {
          CheckShape("AbstractSampler::machine_func_", "out", out.size(),
                     x.size());
          Eigen::Map<Eigen::ArrayXd>{out.data(), out.size()} =
              Eigen::Map<const Eigen::ArrayXcd>{x.data(), x.size()}.abs2();
          // std::transform(x.begin(), x.end(), out.begin(),
          //                [](Complex z) { return std::norm(z); });
        }},
        psi_{psi} {}

 private:
  MachineFunction machine_func_;
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
inline Index CheckBatchSize(const char* func, const Index batch_size) {
  if (batch_size <= 0) {
    std::ostringstream msg;
    msg << func << ": invalid batch size: " << batch_size
        << "; expected a positive number";
    throw InvalidInputError{msg.str()};
  }
  return batch_size;
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

#define NETKET_SAMPLER_APPLY_MACHINE_FUNC(expr)                \
  [this](const Complex z) {                                    \
    double result;                                             \
    this->GetMachineFunc()(nonstd::span<const Complex>{&z, 1}, \
                           nonstd::span<double>{&result, 1});  \
    return result;                                             \
  }(expr)

}  // namespace netket
#endif
