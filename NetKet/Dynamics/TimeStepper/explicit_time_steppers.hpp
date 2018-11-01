#ifndef NETKET_EXPLICIT_TIME_STEPPERS_HPP
#define NETKET_EXPLICIT_TIME_STEPPERS_HPP

#include <array>
#include <cassert>

#include "abstract_time_stepper.hpp"

namespace netket {
namespace ode {

/**
 * Base class for explicit time-stepping methods. This class has
 * an internal time step that may be smaller than the step size of
 * the outer time propagation loop and calls PerformSingleStep
 * on the derived class with the internal step size to perform a
 * single step from t to t + dt.
 */
template <class State>
class ExplicitStepperBase : public AbstractTimeStepper<State> {
  double internal_dt_;

 public:
  void Propagate(OdeSystemFunction<State> ode_system, State &x, double t,
                 double dt) final override {
    assert(dt > .0);
    double tmax = t + dt;

    int step = 0;
    double current_t = t;
    while (current_t < tmax) {
      double next_dt =
          (current_t + internal_dt_ < tmax) ? internal_dt_ : tmax - current_t;

      PerformSingleStep(ode_system, x, current_t, next_dt);

      step++;
      current_t = t + step * internal_dt_;
    };
  }

 protected:
  explicit ExplicitStepperBase(double dt) : internal_dt_(dt) {}

  /**
   * Perform a single step from t to t + dt.
   * @param ode_system The ODE system function.
   * @param x The current state, which is modified in place.
   * @param t The current time.
   * @param dt The time step.
   */
  virtual void PerformSingleStep(OdeSystemFunction<State> ode_system, State &x,
                                 double t, double dt) = 0;
};

/**
 * Implements the explicit Euler method for time stepping.
 * This is only for test purposes. Do not use this class for
 * anything else, as other methods perform better numerically.
 */
template <class State>
class EulerTimeStepper final : public ExplicitStepperBase<State> {
  using Base = ExplicitStepperBase<State>;

  State dxdt_;

 public:
  template <typename Size>
  EulerTimeStepper(double dt, Size state_size) : Base(dt), dxdt_(state_size) {}

 protected:
  void Reset() override {}

  void PerformSingleStep(OdeSystemFunction<State> ode_system, State &x,
                         double t, double dt) override {
    ode_system(x, dxdt_, t);
    x = x + dt * dxdt_;
  }
};

/**
 * Implements the classical 4th order Runge-Kutta method.
 */
template <class State>
class RungeKutta4Stepper final : public ExplicitStepperBase<State> {
  using Base = ExplicitStepperBase<State>;

  std::array<State, 4> k_;
  State x_temp_;

 public:
  template <typename Size>
  RungeKutta4Stepper(double dt, Size state_size)
      : Base(dt)

  {
    for (auto &k : k_) {
      k.resize(state_size);
    }
    x_temp_.resize(state_size);
  }

 protected:
  void Reset() override {}

  void PerformSingleStep(OdeSystemFunction<State> ode_system, State &x,
                         double t, double dt) override {
    const double dt2 = dt / 2.0;
    const double dt6 = dt / 6.0;

    ode_system(x, k_[0], t);

    x_temp_ = x + dt2 * k_[0];
    ode_system(x_temp_, k_[1], t + dt2);

    x_temp_ = x + dt2 * k_[1];
    ode_system(x_temp_, k_[2], t + dt2);

    x_temp_ = x + dt * k_[2];
    ode_system(x_temp_, k_[3], t + dt);

    x = x + dt6 * (k_[0] + 2 * k_[1] + 2 * k_[2] + k_[3]);
  }
};

}  // namespace ode
}  // namespace netket

#endif  // NETKET_EXPLICIT_TIME_STEPPERS_HPP
