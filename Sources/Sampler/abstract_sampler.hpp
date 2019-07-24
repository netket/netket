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
  using MachineFunction = std::function<double(const Complex&)>;

  virtual void Reset(bool initrandom = false) = 0;

  virtual void Sweep() = 0;

  virtual std::pair<Eigen::Ref<const RowMatrix<double>>,
                    Eigen::Ref<const Eigen::VectorXcd>>
  CurrentState() const = 0;

  virtual void SetVisible(Eigen::Ref<const RowMatrix<double>> v) = 0;

  virtual ~AbstractSampler() {}

  void Seed(DistributedRandomEngine::ResultType base_seed) {
    engine_.Seed(base_seed);
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
      : engine_{},
        machine_func_{[](const Complex& z) { return std::norm(z); }},
        psi_{psi} {}

  default_random_engine& GetRandomEngine() { return engine_.Get(); }

 private:
  DistributedRandomEngine engine_;
  MachineFunction machine_func_;
  AbstractMachine& psi_;
};

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

/// \brief A helper class to keep track of the logarithm of the wavefunction.
///
/// It uses Kahan's summation algorithm (Neumaier's adaptation of it) under
/// the hood to reduce accumulation errors for long Markov chains.
class LogValAccumulator {
  using Real = long double;

  struct Accumulator {
    Real sum;
    Real correction;

    /// \brief Adds \p value to the accumulator.
    Accumulator& operator+=(Real value) noexcept {
      const auto t = sum + value;
      correction += std::abs(sum) >= std::abs(value) ? (sum - t) + value
                                                     : (value - t) + sum;
      sum = t;
      return *this;
    }

    /// \brief Conversion to `double` as a way to get the current sum.
    explicit operator double() const noexcept {
      return static_cast<double>(sum + correction);
    }
  };

  Accumulator real_;
  Accumulator imag_;
  mutable Complex result_;

 public:
  explicit LogValAccumulator(const Complex log_val = {0, 0})
      : real_{log_val.real(), 0}, imag_{log_val.imag(), 0}, result_{0, 0} {}

  LogValAccumulator(const LogValAccumulator&) = default;
  LogValAccumulator(LogValAccumulator&&) = default;
  LogValAccumulator& operator=(const LogValAccumulator&) = default;
  LogValAccumulator& operator=(LogValAccumulator&&) = default;

  const Complex& LogVal() const noexcept {
    result_ = Complex{static_cast<double>(real_), static_cast<double>(imag_)};
    return result_;
  }

  LogValAccumulator& operator=(const Complex log_val) noexcept {
    return *this = LogValAccumulator{log_val};
  }

  LogValAccumulator& operator+=(const Complex log_val) noexcept {
    real_ += log_val.real();
    imag_ += log_val.imag();
    return *this;
  }
};

#define NETKET_SAMPLER_SET_VISIBLE_DEFAULT(var)                     \
  void SetVisible(Eigen::Ref<const RowMatrix<double>> v) override { \
    CheckShape(__FUNCTION__, "v", {v.rows(), v.cols()},             \
               {1, GetMachine().Nvisible()});                       \
    var = v.row(0);                                                 \
    Reset(false);                                                   \
  }

#define NETKET_SAMPLER_ACCEPTANCE_DEFAULT(accepts, moves)                  \
  double Acceptance() const {                                              \
    NETKET_CHECK(moves > 0, RuntimeError,                                  \
                 "Cannot compute acceptance, because no moves were made"); \
    return static_cast<double>(accepts) / static_cast<double>(moves);      \
  }

#define NETKET_SAMPLER_ACCEPTANCE_DEFAULT_PT(accepts, moves)               \
  Eigen::VectorXd Acceptance() const {                                     \
    NETKET_CHECK((moves.array() > 0).all(), RuntimeError,                  \
                 "Cannot compute acceptance, because no moves were made"); \
    return (accepts.array() / moves.array()).matrix();                     \
  }

}  // namespace netket
#endif
