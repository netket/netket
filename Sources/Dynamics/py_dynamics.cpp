#include "py_dynamics.hpp"

#include <Eigen/Dense>

#include "TimeStepper/abstract_time_stepper.hpp"
#include "TimeStepper/controlled_time_steppers.hpp"
#include "TimeStepper/explicit_time_steppers.hpp"
#include "Utils/pybind_helpers.hpp"

namespace py = pybind11;

namespace netket {
namespace ode {

namespace detail {

template <class Stepper>
std::unique_ptr<Stepper> ControlledStepperFromKwargs(int dim,
                                                     pybind11::kwargs kwargs) {
  double atol = GetOrDefault(kwargs, "abs_tol", 1e-9);
  double rtol = GetOrDefault(kwargs, "rel_tol", 1e-9);
  return std::unique_ptr<Stepper>(new Stepper(atol, rtol, dim));
}

template <class Stepper>
std::unique_ptr<Stepper> ExplicitStepperFromKwargs(int dim,
                                                   pybind11::kwargs kwargs) {
  double dt = GetOrThrow<double>(kwargs, "internal_dt");
  return std::unique_ptr<Stepper>(new Stepper(dt, dim));
}

}  // namespace detail

template <class State>
std::unique_ptr<AbstractTimeStepper<State>> CreateStepper(
    int dim, const std::string& name, pybind11::kwargs kwargs) {
  if (name == "Dopri54") {
    return detail::ControlledStepperFromKwargs<Dopri54TimeStepper<State>>(
        dim, kwargs);
  } else if (name == "Heun") {
    return detail::ControlledStepperFromKwargs<HeunTimeStepper<State>>(dim,
                                                                       kwargs);
  } else if (name == "RungeKutta4") {
    return detail::ExplicitStepperFromKwargs<RungeKutta4Stepper<State>>(dim,
                                                                        kwargs);
  } else if (name == "Euler") {
    return detail::ExplicitStepperFromKwargs<EulerTimeStepper<State>>(dim,
                                                                      kwargs);
  } else {
    std::stringstream str;
    str << "Unknown TimeStepper: " << name << std::endl;
    throw InvalidInputError(str.str());
  }
}

}  // namespace ode

void AddDynamicsModule(py::module m) {
  auto subm = m.def_submodule("dynamics");

  using State = Eigen::VectorXcd;

  py::class_<ode::AbstractTimeStepper<State>>(subm, "AbstractTimeStepper");

  subm.def("create_timestepper", &ode::CreateStepper<State>, py::arg("dim"),
           py::arg("name") = "Dopri54");
}

}  // namespace netket
