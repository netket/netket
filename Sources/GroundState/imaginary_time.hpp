#ifndef NETKET_IMAGINARY_TIME_PROPAGATION_HPP
#define NETKET_IMAGINARY_TIME_PROPAGATION_HPP

#include <Eigen/Dense>

#include "Dynamics/TimeStepper/abstract_time_stepper.hpp"
#include "Operator/MatrixWrapper/matrix_wrapper.hpp"
#include "Operator/operator.hpp"
#include "Output/json_output_writer.hpp"
#include "Stats/stats.hpp"

namespace netket {

class ImagTimePropagation {
 public:
  using StateVector = Eigen::VectorXcd;
  using Stepper = ode::AbstractTimeStepper<StateVector>;
  using Matrix = AbstractMatrixWrapper<>;

  using ObsEntry = std::pair<std::string, std::unique_ptr<Matrix>>;

  ImagTimePropagation(const AbstractOperator& hamiltonian, Stepper& stepper,
                      double t0, StateVector initial_state,
                      const std::string& matrix_type = "sparse")
      : matrix_(CreateMatrixWrapper<>(hamiltonian, matrix_type)),
        stepper_(stepper),
        t_(t0),
        state_(std::move(initial_state)) {
    ode_system_ = [this](const StateVector& x, StateVector& dxdt,
                         double /*t*/) { dxdt.noalias() = -matrix_->Apply(x); };
  }

  void AddObservable(const AbstractOperator& observable,
                     const std::string& name,
                     const std::string& matrix_type = "sparse") {
    auto wrapper = CreateMatrixWrapper(observable, matrix_type);
    observables_.emplace_back(name, std::move(wrapper));
  }

  void Advance(double dt) {
    // Propagate the state
    stepper_.Propagate(ode_system_, state_, t_, dt);
    // renormalize the state to prevent unbounded growth of the norm
    state_.normalize();
    ComputeObservables(state_);
    t_ += dt;
  }

  void ComputeObservables(const StateVector& state) {
    const auto mean_variance = matrix_->MeanVariance(state);
    obsmanager_.Reset("Energy");
    obsmanager_.Push("Energy", mean_variance[0].real());
    obsmanager_.Reset("EnergyVariance");
    obsmanager_.Push("EnergyVariance", mean_variance[1].real());

    for (const auto& entry : observables_) {
      const auto& name = entry.first;
      const auto& obs = entry.second;
      obsmanager_.Reset(name);

      const auto value = obs->Mean(state).real();
      obsmanager_.Push(name, value);
    }
  }

  const ObsManager& GetObsManager() const { return obsmanager_; }

  double GetTime() const { return t_; }
  void SetTime(double t) { t_ = t; }

 private:
  std::unique_ptr<Matrix> matrix_;
  Stepper& stepper_;
  ode::OdeSystemFunction<StateVector> ode_system_;

  double t_;
  StateVector state_;

  std::vector<ObsEntry> observables_;
  ObsManager obsmanager_;
};

}  // namespace netket

#endif  // NETKET_IMAGINARY_TIME_PROPAGATION_HPP
