//
// Created by Filippo Vicentini on 08/11/2019.
//

#ifndef NETKET_PY_LOCAL_LIOUVILLIAN_HPP
#define NETKET_PY_LOCAL_LIOUVILLIAN_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "local_liouvillian.hpp"
#include "local_operator.hpp"

namespace py = pybind11;

namespace netket {
void AddLocalSuperOperatorModule(py::module &subm);
}
#endif  // NETKET_PY_LOCAL_LIOUVILLIAN_HPP
