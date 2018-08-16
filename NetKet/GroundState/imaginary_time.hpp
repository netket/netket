#ifndef NETKET_IMAGINARY_TIME_PROPAGATION_HPP
#define NETKET_IMAGINARY_TIME_PROPAGATION_HPP

#include <Eigen/Dense>

#include "Hamiltonian/MatrixWrapper/matrix_wrapper.hpp"
#include "Observable/observable.hpp"
#include "Stats/stats.hpp"
#include "UnitaryDynamics/time_evolution.hpp"

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

  static ImaginaryTimePropagation FromJson(const Hamiltonian& hamiltonian,
                                           const std::vector<Observable>& observables,
                                           const json& pars) {
    auto matrix = ConstructMatrixWrapper(pars, hamiltonian);
    auto stepper =
        ode::ConstructTimeStepper<State>(pars, matrix->GetDimension());
    auto range = TimeRange::FromJson(pars);

    ObservableVector wrapped_observables;
    for (const auto& obs : observables) {
      wrapped_observables.emplace_back(obs.Name(),
                                       ConstructMatrixWrapper(pars, obs));
    }

    const std::string filebase = FieldVal(pars, "OutputFile");
    std::ofstream outstream{filebase + ".log"};

    return ImaginaryTimePropagation(std::move(matrix), std::move(stepper),
                                    std::move(wrapped_observables),
                                    std::move(outstream), range);
  }

  ImaginaryTimePropagation(std::unique_ptr<Matrix> matrix,
                           std::unique_ptr<Stepper> stepper,
                           ObservableVector observables, std::ofstream filelog,
                           TimeRange time_range)
      : matrix_(std::move(matrix)),
        stepper_(std::move(stepper)),
        observables_(std::move(observables)),
        filelog_(std::move(filelog)),
        range_(time_range) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
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
      PrintOutput(step, t);

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

  void PrintOutput(int i, double time) {
    auto jiter = json(obsmanager_);
    jiter["Iteration"] = i;
    jiter["Time"] = time;
    outputjson_["Output"].push_back(jiter);

    if (mpi_rank_ == 0) {
      if (jiter["Iteration"] != 0) {
        long pos = filelog_.tellp();
        filelog_.seekp(pos - 3);
        filelog_.write(",  ", 3);
        filelog_ << jiter << "]}" << std::endl;
      } else {
        filelog_ << outputjson_ << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  int GetDimension() const { return matrix_->GetDimension(); }

 private:
  std::unique_ptr<Matrix> matrix_;
  std::unique_ptr<Stepper> stepper_;
  ode::OdeSystemFunction<State> ode_system_;

  ObservableVector observables_;
  ObsManager obsmanager_;
  json outputjson_;
  std::ofstream filelog_;

  int mpi_rank_;

  TimeRange range_;
};

}  // namespace netket

#endif  // NETKET_IMAGINARY_TIME_PROPAGATION_HPP
