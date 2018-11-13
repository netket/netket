#ifndef NETKET_IMAGINARY_TIME_PROPAGATION_HPP
#define NETKET_IMAGINARY_TIME_PROPAGATION_HPP

#include <Eigen/Dense>

#include "Dynamics/TimeStepper/time_stepper.hpp"
#include "Operator/MatrixWrapper/matrix_wrapper.hpp"
#include "Operator/observable.hpp"
#include "Stats/stats.hpp"

#include "json_output_writer.hpp"

// TODO remove Observable and replace with AbstractOperator+name
// Provide a method AddObservable, as in VariationalMonteCarlo
namespace netket {

class ImaginaryTimeDriver {
 public:
  using State = Eigen::VectorXcd;
  using Stepper = ode::AbstractTimeStepper<State>;
  using Matrix = AbstractMatrixWrapper<>;

  using ObsEntry = std::pair<std::string, std::unique_ptr<Matrix>>;
  using ObservableVector = std::vector<ObsEntry>;

  ImaginaryTimeDriver(Matrix& matrix, Stepper& stepper,
                      JsonOutputWriter& output, double tmin, double tmax,
                      double dt)
      : matrix_(matrix),
        stepper_(stepper),
        output_(output),
        range_(ode::TimeRange{tmin, tmax, dt}) {
    ode_system_ = [this](const State& x, State& dxdt, double /*t*/) {
      dxdt.noalias() = -matrix_.Apply(x);
    };
  }

  void AddObservable(const AbstractOperator& observable,
                     const std::string& name,
                     const std::string& matrix_type = "Sparse") {
    auto wrapper = CreateMatrixWrapper(observable, matrix_type);
    observables_.emplace_back(name, std::move(wrapper));
  }

  void Run(State& initial_state) {
    assert(initial_state.size() == Dimension());

    int step = 0;
    double t = range_.tmin;
    while (t < range_.tmax) {
      double next_dt =
          (t + range_.dt <= range_.tmax) ? range_.dt : range_.tmax - t;

      stepper_.Propagate(ode_system_, initial_state, t, next_dt);

      // renormalize the state to prevent unbounded growth of the norm
      initial_state.normalize();

      ComputeObservables(initial_state);
      auto obs_data = json(obsmanager_);
      output_.WriteLog(step, obs_data, t);
      output_.WriteState(step, initial_state);

      step++;
      t = range_.tmin + step * range_.dt;
    };
  }

  void ComputeObservables(const State& state) {
    const auto mean_variance = matrix_.MeanVariance(state);
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

  int Dimension() const { return matrix_.Dimension(); }

 private:
  Matrix& matrix_;
  Stepper& stepper_;
  ode::OdeSystemFunction<State> ode_system_;

  ObservableVector observables_;
  ObsManager obsmanager_;

  JsonOutputWriter& output_;

  ode::TimeRange range_;
};

}  // namespace netket

#endif  // NETKET_IMAGINARY_TIME_PROPAGATION_HPP
