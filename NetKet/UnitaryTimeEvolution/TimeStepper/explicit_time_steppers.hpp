#ifndef NETKET_EXPLICIT_TIME_STEPPERS_HPP
#define NETKET_EXPLICIT_TIME_STEPPERS_HPP

#include <array>
#include <cassert>

#include "abstract_time_stepper.hpp"

namespace netket { namespace ode {

/**
 * Base class for explicit time-stepping methods. This class has
 * an internal time step that may be smaller than the step size of
 * the outer time propagation loop and calls PerformSingleStep
 * on the derived class with the internal step size to perform a
 * single step from t to t + dt.
 *
 * This class uses the CRTP (i.e., using the derived class as
 * template parameter) in order to call the dervied class' methods
 * without needing a virtual function lookup.
 */
template<class Derived, class State, typename Time>
class ExplicitStepperBase : public AbstractTimeStepper<State, Time>
{
    Time internal_dt_;

public:
    void Propagate(OdeSystemFunction<State> ode_system,
                   State &x,
                   Time t, Time dt) override
    {
        assert(dt > .0);
        double tmax = t + dt;

        int step = 0;
        double current_t = t;
        while(current_t < tmax)
        {
            double next_dt = (current_t + internal_dt_ < tmax) ? internal_dt_ : tmax - current_t;

            PerformSingleStepImpl(ode_system, x, current_t, next_dt);

            step++;
            current_t = t + step * internal_dt_;
        };
    }

protected:
    explicit ExplicitStepperBase(Time dt)
            : internal_dt_(dt)
    {
    }

    void PerformSingleStepImpl(OdeSystemFunction<State> ode_system,
                           State &x,
                           Time t, Time dt)
    {
        static_cast<Derived*>(this)->PerformSingleStep(ode_system, x, t, dt);
    }
};

/**
 * Implements the explicit Euler method for time stepping.
 * This is only for test purposes. Do not use this class for
 * anything else, as other methods perform better numerically.
 */
template<class State, typename Time = double>
class EulerTimeStepper : public ExplicitStepperBase<EulerTimeStepper<State, Time>, State, Time>
{
    using Base = ExplicitStepperBase<EulerTimeStepper, State, Time>;

    State dxdt_;

public:
    template<typename Size>
    EulerTimeStepper(Time dt, Size state_size)
            : Base(dt),
              dxdt_(state_size)
    {
    }

    void PerformSingleStep(OdeSystemFunction <State> &ode_system,
                            State &x,
                            double t, double dt)
    {
        ode_system(x, dxdt_, t);
        x = x + dt * dxdt_;
    }
};

/**
 * Implements the classical 4th order Runge-Kutta method.
 */
template<class State, typename Time = double>
class RungeKutta4Stepper : public ExplicitStepperBase<RungeKutta4Stepper<State, Time>, State, Time>
{
    using Base = ExplicitStepperBase<RungeKutta4Stepper, State, Time>;

    std::array<State, 4> k_;
    State x_temp_;

public:
    template<typename Size>
    RungeKutta4Stepper(Time dt, Size state_size)
            : Base(dt)

    {
        for(auto& k : k_)
        {
            k.resize(state_size);
        }
        x_temp_.resize(state_size);
    }

    void PerformSingleStep(OdeSystemFunction <State> &ode_system,
                           State &x,
                           double t, double dt)
    {
        const double dt2 = dt / 2.0;
        const double dt6 = dt / 6.0;

        ode_system(x, k_[0], t);

        x_temp_ = x + dt2 * k_[0];
        ode_system(x_temp_, k_[1], t + dt2);

        x_temp_ = x + dt2 * k_[1];
        ode_system(x_temp_, k_[2], t + dt2);

        x_temp_ = x + dt * k_[2];
        ode_system(x_temp_, k_[3], t + dt);

        x = x + dt6 * (k_[0] + 2*k_[1] + 2*k_[2] + k_[3]);
    }
};

}}

#endif //NETKET_EXPLICIT_TIME_STEPPERS_HPP
