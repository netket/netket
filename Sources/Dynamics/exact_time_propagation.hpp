// Copyright 2018-2019 The Simons Foundation, Inc. - All Rights Reserved.
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

#ifndef NETKET_EXACT_TIME_PROPAGATION_HPP
#define NETKET_EXACT_TIME_PROPAGATION_HPP

#include <unordered_map>

#include <Eigen/Dense>

#include "Dynamics/TimeStepper/abstract_time_stepper.hpp"
#include "Operator/operator.hpp"
#include "Output/json_output_writer.hpp"
#include "Stats/binning.hpp"

namespace netket {

class ExactTimePropagation {
 public:
  using StateVector = Eigen::VectorXcd;
  using Stepper = ode::AbstractTimeStepper<StateVector>;
  using Matrix = std::function<StateVector(const StateVector&)>;
  using StatsMap = std::unordered_map<std::string, Binning<double>::Stats>;

  static Matrix MakeMatrix(const AbstractOperator& op,
                           const std::string& type) {
    if (type == "dense") {
      struct Function {
        Eigen::MatrixXcd matrix;
        StateVector operator()(const StateVector& x) const {
          return matrix * x;
        }
      };
      return Function{op.ToDense()};
    } else if (type == "direct") {
      struct Function {
        const AbstractOperator& matrix;
        StateVector operator()(const StateVector& x) const {
          return matrix.Apply(x);
        }
      };
      return Function{op};
    } else if (type == "sparse") {
      struct Function {
        Eigen::SparseMatrix<Complex> matrix;
        StateVector operator()(const StateVector& x) const {
          return matrix * x;
        }
      };
      return Function{op.ToSparse()};
    } else {
      std::stringstream str;
      str << "Unknown matrix type: " << type;
      throw InvalidInputError(str.str());
    }
  }

  ExactTimePropagation(const AbstractOperator& hamiltonian, Stepper& stepper,
                       double t0, StateVector initial_state,
                       const std::string& matrix_type = "sparse",
                       const std::string& propagation_type = "exact")
      : matrix_{MakeMatrix(hamiltonian, matrix_type)},
        stepper_(stepper),
        t_(t0),
        state_(std::move(initial_state)) {
    if (propagation_type == "imaginary") {
      ode_system_ = [this](const StateVector& x, StateVector& dxdt,
                           double /*t*/) { dxdt.noalias() = -matrix_(x); };
      normalize_ = true;
    } else if (propagation_type == "real") {
      static constexpr const Complex mi{0, -1};
      ode_system_ = [this](const StateVector& x, StateVector& dxdt,
                           double /*t*/) { dxdt.noalias() = mi * matrix_(x); };
      normalize_ = false;
    } else {
      throw InvalidInputError{
          "ExactTimePropagation: propagation_type must be 'real' or "
          "'imaginary'."};
    }
  }

  void AddObservable(const AbstractOperator& observable,
                     const std::string& name,
                     const std::string& matrix_type = "sparse") {
    observables_.emplace_back(name, MakeMatrix(observable, matrix_type));
  }

  void Advance(double dt) {
    // Propagate the state
    stepper_.Propagate(ode_system_, state_, t_, dt);
    if (normalize_) {
      // Renormalize state to prevent unbounded growth in imaginary time
      // propagation
      state_.normalize();
    }
    ComputeObservables(state_);
    t_ += dt;
  }

  void ComputeObservables(const StateVector& state) {
    const auto mean_variance = MeanVariance(matrix_, state);
    observable_stats_["Energy"] = {mean_variance.first.real(), 0.0, 0.0};
    observable_stats_["EnergyVariance"] = {mean_variance.second, 0.0, 0.0};

    for (const auto& entry : observables_) {
      const auto& name = entry.first;
      const auto& obs = entry.second;
      observable_stats_[name] = {Mean(obs, state).real(), 0.0, 0.0};
    }
  }

  const StatsMap& GetObservableStats() const noexcept { return observable_stats_; }

  double GetTime() const noexcept { return t_; }
  void SetTime(double t) { t_ = t; }

  const StateVector &GetState() const noexcept { return state_; }
  void SetState(const StateVector &state) { state_ = state; }

 private:
  Matrix matrix_;
  Stepper& stepper_;
  ode::OdeSystemFunction<StateVector> ode_system_;
  bool normalize_;

  double t_;
  StateVector state_;

  using ObsEntry = std::pair<std::string, Matrix>;
  std::vector<ObsEntry> observables_;
  StatsMap observable_stats_;
};

}  // namespace netket

#endif  // NETKET_EXACT_TIME_PROPAGATION_HPP
