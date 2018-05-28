#ifndef NETKET_CONTROLLED_TIME_STEPPERS_HPP
#define NETKET_CONTROLLED_TIME_STEPPERS_HPP

#include <array>
#include <cassert>
#include <cmath>

#include "abstract_time_stepper.hpp"

namespace netket { namespace ode {

/**
 * Base class for error-controlled time-stepping methods.
 *
 * This class uses the CRTP (i.e., using the derived class as
 * template parameter) in order to call the dervied class methods
 * without needing a virtual function lookup.
 */
template<class Derived, class State, typename Time>
class ControlledStepperBase
        : AbstractTimeStepper<State, Time>
{
    const double atol_;
    const double rtol_;

    State last_x_;
    double last_norm_;

    static constexpr double safety_factor = 0.95;
    static constexpr double err_exponent = -1. / 5.;

public:
    void Propagate(OdeSystemFunction<State> ode_system,
                   State &x,
                   Time t, Time dt) override
    {
        assert(dt > .0);
        double tmax = t + dt;

        last_x_.resize(x.size());
        last_x_ = x;
        last_norm_ = x.norm();

        int step = 0;
        double current_t = t;
        double current_dt = dt / 10.; // TODO: better way to guess initial step size?
        while(current_t < tmax)
        {
            double next_dt = (current_t + current_dt < tmax) ? current_dt : tmax - current_t;

            double scaled_error = PerformSingleStepImpl(ode_system, x, current_t, next_dt);

            if(scaled_error <= 1.0) // error is within bounds
            {
                step++;
                current_t += next_dt;
                last_x_ = x;
                last_norm_ = x.norm();
            }
            else // error too large, try again with new step size
            {
                x = last_x_;
            }

            // update time step
            double dt_factor = safety_factor * std::pow(scaled_error, err_exponent);
            dt_factor = std::clamp(dt_factor, 0.01, 10.);
            current_dt *= dt_factor;
        };
    }

protected:
    explicit ControlledStepperBase(double atol, double rtol)
            : atol_(atol), rtol_(rtol)
    {
    }

    double PerformSingleStepImpl(OdeSystemFunction<State> ode_system,
                       State &x,
                       Time t, Time dt)
    {
        return static_cast<Derived*>(this)->PerformSingleStep(ode_system, x, t, dt);
    }

    double LocalError(const State& delta, double norm) const
    {
        assert(norm >= .0);

        // use maximum of last norms in case one is close to zero
        double norm1 = std::max(norm, last_norm_);

        double scale = (atol_ + norm1 * rtol_) * std::sqrt(delta.size());
        return delta.norm() / scale;
    }
};

/**
 * Implements the Heun method for time-stepping.
 * This is a controlled second-order Runge-Kutta method with an
 * embedded first-order methods (which is just Euler) for error estimation.
 */
template<class State, typename Time = double>
class HeunTimeStepper
        : public ControlledStepperBase<HeunTimeStepper<State, Time>, State, Time>
{
    using Base = ControlledStepperBase<HeunTimeStepper, State, Time>;

    State k1_;
    State k2_;
    State x_temp_;

public:
    template<typename Size>
    HeunTimeStepper(double atol, double rtol, Size state_size)
            : Base(atol, rtol),
              k1_(state_size),
              k2_(state_size),
              x_temp_(state_size)
    {
    }

    double PerformSingleStep(OdeSystemFunction <State> &ode_system,
                             State &x, double t, double dt)
    {
        //std::cout << "Heun    " << "t=" << t << "\t\tcurr_dt=" << dt << std::endl;

        ode_system(x, k1_, t);

        x_temp_ = x + dt * k1_;
        ode_system(x_temp_, k2_, t + dt);

        x = x + (dt/2.0) * (k1_ + k2_);

        return Base::LocalError(x - x_temp_, x.norm());
    }
};

/**
 * Implements the Dormand-Prince 5th order method for time-stepping.
 * This is a controlled Runge-Kutta method with embedded an fourth-order
 * scheme for error estimation.
 */
template<class State, typename Time = double>
class Dopri54TimeStepper
        : public ControlledStepperBase<Dopri54TimeStepper<State, Time>, State, Time>
{
    using Base = ControlledStepperBase<Dopri54TimeStepper, State, Time>;

    std::array<State, 7> k_;
    State x_temp1_;
    State delta_;

    /* Butcher tableau */
    static const int N_ = 7;

    static constexpr double A_[N_][N_-1] = {
        .0,           .0,          .0,           .0,         .0,          .0,
        1./5,         .0,          .0,           .0,         .0,          .0,
        3./40,        9./40,       .0,           .0,         .0,          .0,
        44./45,      -56./15,      32./9,        .0,         .0,          .0,
        19372./6561, -25360./2187, 64448./6561, -212./729,   .0,          .0,
        9017./3168,  -355./33,     46732./5247,  49./176,   -5103./18656, .0,
        35./384,      .0,          500./1113,    125./192,  -2187./6784,  11./84
    };

    // coefficients for 5th-order scheme
    static constexpr double BA_[N_] = {35./384, .0, 500./1113,
                                      125./192, -2187./6784, 11./84, .0};
    // coefficients for embedded 4th-order scheme
    static constexpr double BB_[N_] = {5179./57600, .0, 7571./16695, 393./640,
                                       -92097./339200, 187./2100, 1./40};

    static constexpr double C_[N_] = {.0, 1./5, 3./10, 4./5, 8./9, 1., 1.};

public:
    template<typename Size>
    Dopri54TimeStepper(double atol, double rtol, Size state_size)
            : Base(atol, rtol),
              x_temp1_(state_size)
    {
        for(auto &k : k_)
        {
            k.resize(state_size);
        }
    }

    double PerformSingleStep(OdeSystemFunction <State> &ode_system,
                             State &x, double t, double dt)
    {
        //std::cout << "Dopri54 " << "t=" << t << "\t\tcurr_dt=" << dt << std::endl;

        ode_system(x, k_[0], t);
        for(int i = 1; i < N_; i++)
        {
            x_temp1_ = x + dt * A_[i][0] * k_[0];
            for(int j = 1; j < i; j++)
            {
                x_temp1_ += dt * A_[i][j] * k_[j];
            }
            ode_system(x_temp1_, k_[i], t + C_[i] * dt);
        }

        x = x_temp1_;

        delta_ = dt * (BA_[0] - BB_[0]) * k_[0];
        for(int i = 1; i < N_; i++)
        {
            delta_ += dt * (BA_[i] - BB_[i]) * k_[i];
        }

        return Base::LocalError(delta_, x.norm());
    }
};

}}

#endif //NETKET_CONTROLLED_TIME_STEPPERS_HPP
