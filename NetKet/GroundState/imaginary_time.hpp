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
  class Iterator;

  using StateVector = Eigen::VectorXcd;
  using Stepper = ode::AbstractTimeStepper<StateVector>;
  using Matrix = AbstractMatrixWrapper<>;

  using ObsEntry = std::pair<std::string, std::unique_ptr<Matrix>>;

  ImagTimePropagation(Matrix& matrix, Stepper& stepper, double t0,
                      StateVector initial_state)
      : matrix_(matrix),
        stepper_(stepper),
        t_(t0),
        state_(std::move(initial_state)) {
    ode_system_ = [this](const StateVector& x, StateVector& dxdt,
                         double /*t*/) { dxdt.noalias() = -matrix_.Apply(x); };
  }

  void AddObservable(const AbstractOperator& observable,
                     const std::string& name,
                     const std::string& matrix_type = "Sparse") {
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

  /*void Run(StateVector& initial_state, ) {
    assert(initial_state.size() == Dimension());
    state_ = initial_state;
    for (const auto& step : Iterate(range.t0, )) {
    }
  }*/

  Iterator Iterate(double dt,
                   nonstd::optional<Index> max_steps = nonstd::nullopt,
                   bool store_state = true) {
    return Iterator(*this, dt, std::move(max_steps), store_state);
  }

  void ComputeObservables(const StateVector& state) {
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

  double GetTime() const { return t_; }
  void SetTime(double t) { t_ = t; }

  /**
   * Struct storing information on a single ITP step.
   */
  struct Step {
    Index index;
    double t;
    ObsManager observables;
    nonstd::optional<StateVector> state;
  };

  class Iterator {
   public:
    // typedefs required for iterators
    using iterator_category = std::input_iterator_tag;
    using difference_type = Index;
    using value_type = Step;
    using pointer_type = Step*;
    using reference_type = Step&;

   private:
    ImagTimePropagation& driver_;
    nonstd::optional<Index> max_iter_;
    double dt_;
    bool store_state_;

    Index cur_iter_;

   public:
    Iterator(ImagTimePropagation& driver, double dt,
             nonstd::optional<Index> max_iter, bool store_state)
        : driver_(driver),
          max_iter_(std::move(max_iter)),
          dt_(dt),
          store_state_(store_state),
          cur_iter_(0) {}

    Step operator*() const {
      using OptionalVec = nonstd::optional<Eigen::VectorXcd>;
      auto state = store_state_ ? OptionalVec(driver_.state_) : nonstd::nullopt;
      return {cur_iter_, driver_.t_, driver_.obsmanager_, std::move(state)};
    };
    Iterator& operator++() {
      driver_.Advance(dt_);
      cur_iter_ += 1;
      return *this;
    }

    // TODO(C++17): Replace with comparison to special Sentinel type, since
    // C++17 allows end() to return a different type from begin().
    bool operator!=(const Iterator&) {
      return !max_iter_.has_value() || cur_iter_ < max_iter_.value();
    }
    bool operator==(const Iterator& other) { return !(*this != other); }

    Iterator begin() const { return *this; }
    Iterator end() const { return *this; }
  };

 private:
  Matrix& matrix_;
  Stepper& stepper_;
  ode::OdeSystemFunction<StateVector> ode_system_;

  double t_;
  StateVector state_;

  std::vector<ObsEntry> observables_;
  ObsManager obsmanager_;
};

}  // namespace netket

#endif  // NETKET_IMAGINARY_TIME_PROPAGATION_HPP
