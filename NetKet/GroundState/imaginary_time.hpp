#ifndef NETKET_IMAGINARY_TIME_PROPAGATION_HPP
#define NETKET_IMAGINARY_TIME_PROPAGATION_HPP

#include <Eigen/Dense>

#include "Hamiltonian/MatrixWrapper/matrix_wrapper.hpp"
#include "Observable/observable.hpp"
#include "Stats/stats.hpp"
#include "UnitaryDynamics/time_evolution.hpp"

#include "json_output_writer.hpp"

namespace netket {

// TODO: Move to UnitaryDynamics
struct TimeRange {
  double tmin;
  double tmax;
  double dt;

  static TimeRange FromJson(const json& pars) {
    double tmin = FieldVal(pars, "StartTime");
    double tmax = FieldVal(pars, "EndTime");
    double dt = FieldVal(pars, "TimeStep");
    return {tmin, tmax, dt};
  }
};

class ImaginaryTimePropagation {
 public:
  using State = Eigen::VectorXcd;
  using Matrix = AbstractMatrixWrapper<Hamiltonian>;
  using Stepper = ode::AbstractTimeStepper<State>;

  using MatrixObs = AbstractMatrixWrapper<Observable>;
  using ObsEntry = std::pair<std::string, std::unique_ptr<MatrixObs>>;
  using ObservableVector = std::vector<ObsEntry>;

  static ImaginaryTimePropagation FromJson(
      const Hamiltonian& hamiltonian,
      const std::vector<Observable>& observables, const json& pars) {
    auto matrix = ConstructMatrixWrapper(pars, hamiltonian);
    auto stepper =
        ode::ConstructTimeStepper<State>(pars, matrix->GetDimension());
    auto range = TimeRange::FromJson(pars);

    ObservableVector wrapped_observables;
    for (const auto& obs : observables) {
      wrapped_observables.emplace_back(obs.Name(),
                                       ConstructMatrixWrapper(pars, obs));
    }

    auto output = JsonOutputWriter::FromJson(pars);

    return ImaginaryTimePropagation(std::move(matrix), std::move(stepper),
                                    std::move(wrapped_observables),
                                    std::move(output), range);
  }

  ImaginaryTimePropagation(std::unique_ptr<Matrix> matrix,
                           std::unique_ptr<Stepper> stepper,
                           ObservableVector observables,
                           JsonOutputWriter output, TimeRange time_range)
      : matrix_(std::move(matrix)),
        stepper_(std::move(stepper)),
        observables_(std::move(observables)),
        output_(std::move(output)),
        range_(time_range) {
    ode_system_ = [this](const State& x, State& dxdt, double /*t*/) {
      dxdt.noalias() = -matrix_->Apply(x);
    };
  }

  void Run(State& state) {
    assert(state.size() == GetDimension());

    int step = 0;
    double t = range_.tmin;
    while (t < range_.tmax) {
      double next_dt =
          (t + range_.dt <= range_.tmax) ? range_.dt : range_.tmax - t;

      stepper_->Propagate(ode_system_, state, t, next_dt);

      // renormalize the state to prevent unbounded growth of the norm
      state.normalize();

      ComputeObservables(state);
      output_.WriteLog(step, obsmanager_, t);
      output_.WriteState(step, state);

      step++;
      t = range_.tmin + step * range_.dt;
    };
  }

  void ComputeObservables(const State& state) {
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

  int GetDimension() const { return matrix_->GetDimension(); }

 private:
  std::unique_ptr<Matrix> matrix_;
  std::unique_ptr<Stepper> stepper_;
  ode::OdeSystemFunction<State> ode_system_;

  ObservableVector observables_;
  ObsManager obsmanager_;

  JsonOutputWriter output_;

  TimeRange range_;
};

}  // namespace netket

#endif  // NETKET_IMAGINARY_TIME_PROPAGATION_HPP
