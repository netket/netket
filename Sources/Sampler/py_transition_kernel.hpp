// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_PY_TRANSITION_KERNEL_HPP
#define NETKET_PY_TRANSITION_KERNEL_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "Utils/memory_utils.hpp"
#include "Utils/parallel_utils.hpp"
#include "Utils/pybind_helpers.hpp"

namespace py = pybind11;

namespace netket {

void AddTransitionKernels(py::module &subm) {
  py::class_<ExchangeKernel>(subm, "ExchangeKernel")
      .def(py::init<const AbstractHilbert &, Index>(), py::arg("hilbert"),
           py::arg("d_max"))
      .def("apply", &ExchangeKernel::operator(), py::arg("v"),
           py::arg("v_prime"), py::arg("log_acceptance_correction"));

  py::class_<HopKernel>(subm, "HopKernel")
      .def(py::init<const AbstractHilbert &, Index>(), py::arg("hilbert"),
           py::arg("d_max"))
      .def("apply", &HopKernel::operator(), py::arg("v"), py::arg("v_prime"),
           py::arg("log_acceptance_correction"));

  py::class_<CustomLocalKernel>(subm, "CustomLocalKernel")
      .def(py::init<const LocalOperator &, const std::vector<double> &>(),
           py::arg("move_operators"), py::arg("move_weights"))
      .def("apply", &CustomLocalKernel::operator(), py::arg("v"),
           py::arg("v_prime"), py::arg("log_acceptance_correction"));
}
}  // namespace netket

#endif
