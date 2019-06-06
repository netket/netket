#ifndef NETKET_DYNAMICS_PY_DYNAMICS_HPP
#define NETKET_DYNAMICS_PY_DYNAMICS_HPP

#include <pybind11/pybind11.h>

namespace netket {

void AddDynamicsModule(pybind11::module m);

}  // namespace netket

#endif  // NETKET_DYNAMICS_PY_DYNAMICS_HPP
