#ifndef TIME_STEPPER_HPP
#define TIME_STEPPER_HPP

#include <memory>

#include "Utils/json_helper.hpp"

#include "explicit_time_steppers.hpp"
#include "controlled_time_steppers.hpp"

namespace netket { namespace ode {

namespace detail {

template<class Stepper>
std::unique_ptr<Stepper> ControlledStepperFromJson(const json& pars, int dim)
{
    double atol = FieldOrDefaultVal(pars, "AbsTol", 1e-9);
    double rtol = FieldOrDefaultVal(pars, "RelTol", 1e-9);
    return std::unique_ptr<Stepper>(new Stepper(atol, rtol, dim));
}

template<class Stepper>
std::unique_ptr<Stepper> ExplicitStepperFromJson(const json& pars, int dim)
{
    double dt = FieldVal(pars, "InternalTimeStep");
    return std::unique_ptr<Stepper>(new Stepper(dt, dim));
}

}

template<class State>
std::unique_ptr<AbstractTimeStepper<State>>
ConstructTimeStepper(const json& pars, int dim)
{
    std::string stepper_name = FieldOrDefaultVal<json, std::string>(pars, "TimeStepper", "Dopri54");
    if(stepper_name == "Dopri54")
    {
        return detail::ControlledStepperFromJson<Dopri54TimeStepper<State>>(pars, dim);
    }
    else if(stepper_name == "Heun")
    {
        return detail::ControlledStepperFromJson<HeunTimeStepper<State>>(pars, dim);
    }
    else if(stepper_name == "RungeKutta4")
    {
        return detail::ExplicitStepperFromJson<RungeKutta4Stepper<State>>(pars, dim);
    }
    else if(stepper_name == "Euler")
    {
        return detail::ExplicitStepperFromJson<EulerTimeStepper<State>>(pars, dim);
    }
    else
    {
        std::cout << "Unknown TimeStepper: " << stepper_name << std::endl;
        std::abort();
    }
}

}}

#endif // TIME_STEPPER_HPP
