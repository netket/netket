#ifndef NETKET_ABSTRACT_TIME_STEPPER_HPP
#define NETKET_ABSTRACT_TIME_STEPPER_HPP

#include <functional>
#include <iostream>

namespace netket { namespace ode
{

/**
 * A function compatible with AbstractTimeStepper::ObserverFunction that does nothing.
 * To be used as default value.
 */
template<class State, typename Time = double>
void NullObserver(const State& /* x */, Time /* t */)
{
    // do nothing
}

template<class OdeState>
using OdeSystemFunction = std::function<void(const OdeState&, OdeState&, double)>;

template<class OdeState>
using ObserverFunction = std::function<void(const OdeState&, double)>;

/**
 * Represents a class to perform ODE time steps.
 * @tparam State The ODE state. Should be an Eigen dynamic-size matrix or
 *      a compatible type.
 * @tparam Time Type of the time variable. Should be float or double.
 */
template<class State, typename Time = double>
class AbstractTimeStepper
{
public:
    virtual void Propagate(OdeSystemFunction<State> ode_system,
                           State &x,
                           Time t, Time dt) = 0;

    virtual ~AbstractTimeStepper() = default;
};

/**
 * Propagates the initial state in time by solving the initial value problem
 *      dx/dt = F(x, t),    x(t0) = initial_state.
 * @param ode_system A callable representing the right-hand side F of the ODE.
 *      The function is called with the current state as const reference, a
 *      reference to the derivative dxdt which ode_system should write to and
 *      the current time t.
 * @param initial_state The value of x at t0.
 * @param t0 The starting time.
 * @param tmax The end time of the propagation. If at time t the next step t + dt
 *      is > tmax, the state is only propagated to tmax.
 * @param dt The time step. Time steppers may use a smaller time step internally
 *      (e.g., as part of an adaptive algorithm).
 * @param observer This callable is called at each time step, i.e., at times
 *      t0, t0 + dt, t0 + 2*dt, ..., tmax.
 */
template<class Stepper, class State, typename Time = double>
void Integrate(Stepper& stepper,
               OdeSystemFunction<State> ode_system,
               State &state,
               Time t0, Time tmax, Time dt,
               ObserverFunction<State> observer = NullObserver<State, Time>)
{
    assert(t0 < tmax);
    assert(dt > .0);

    observer(state, t0);

    int step = 0;
    double t = t0;
    while(t < tmax)
    {
        double next_dt = (t + dt <= tmax) ? dt : tmax - t;
        stepper.Propagate(ode_system, state, t, next_dt);

        step++;
        t = t0 + step * dt;
        observer(state, t);
    };
}

}}

#endif //NETKET_ABSTRACT_TIME_STEPPER_HPP
