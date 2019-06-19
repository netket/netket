#ifndef NETKET_IMAGINARY_TIME_PROPAGATION_HPP
#define NETKET_IMAGINARY_TIME_PROPAGATION_HPP

#include <Eigen/Dense>

#include "Dynamics/TimeStepper/abstract_time_stepper.hpp"
#include "Operator/operator.hpp"
#include "Output/json_output_writer.hpp"
#include "Stats/stats.hpp"

namespace netket {

class ImagTimePropagation {
 public:
  using StateVector = Eigen::VectorXcd;
  using Stepper = ode::AbstractTimeStepper<StateVector>;
  using Operator = std::function<StateVector(const StateVector&)>;
  using ObsEntry = std::pair<std::string, Operator>;

  static Operator MakeOperator(const AbstractOperator& op,
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
      str << "Unknown matrix wrapper: " << type;
      throw InvalidInputError(str.str());
    }
  }

  ImagTimePropagation(const AbstractOperator& hamiltonian, Stepper& stepper,
                      double t0, StateVector initial_state,
                      const std::string& matrix_type = "sparse")
      : matrix_{MakeOperator(hamiltonian, matrix_type)},
        stepper_(stepper),
        t_(t0),
        state_(std::move(initial_state)) {
    ode_system_ = [this](const StateVector& x, StateVector& dxdt,
                         double /*t*/) { dxdt.noalias() = -matrix_(x); };
  }

  void AddObservable(const AbstractOperator& observable,
                     const std::string& name,
                     const std::string& matrix_type = "sparse") {
    observables_.emplace_back(name, MakeOperator(observable, matrix_type));
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
    const auto mean_variance = MeanVariance(matrix_, state);
    obsmanager_.Reset("Energy");
    obsmanager_.Push("Energy", mean_variance.first.real());
    obsmanager_.Reset("EnergyVariance");
    obsmanager_.Push("EnergyVariance", mean_variance.second);

    for (const auto& entry : observables_) {
      const auto& name = entry.first;
      const auto& obs = entry.second;
      obsmanager_.Reset(name);

      const auto value = Mean(obs, state).real();
      obsmanager_.Push(name, value);
    }
  }

  const ObsManager& GetObsManager() const { return obsmanager_; }

  double GetTime() const { return t_; }
  void SetTime(double t) { t_ = t; }

 private:
  Operator matrix_;
  Stepper& stepper_;
  ode::OdeSystemFunction<StateVector> ode_system_;

  double t_;
  StateVector state_;

  std::vector<ObsEntry> observables_;
  ObsManager obsmanager_;
};

}  // namespace netket

#endif  // NETKET_IMAGINARY_TIME_PROPAGATION_HPP
