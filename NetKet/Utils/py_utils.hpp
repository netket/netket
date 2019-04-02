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

#ifndef NETKET_PYUTILS_HPP
#define NETKET_PYUTILS_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "all_utils.hpp"

namespace py = pybind11;

namespace netket {

void AddUtilsModule(py::module &m) {
  auto subm = m.def_submodule("utils");

  py::class_<netket::default_random_engine>(subm, "RandomEngine")
      .def(py::init<netket::default_random_engine::result_type>(),
           py::arg("seed") = netket::default_random_engine::default_seed)
      .def("seed", (void (netket::default_random_engine::*)(
                       netket::default_random_engine::result_type)) &
                       netket::default_random_engine::seed);

  py::class_<Lookup<double>>(m, "LookupReal").def(py::init<>());

  py::class_<Lookup<Complex>>(m, "LookupComplex")
      .def(py::init<>());

  py::class_<MPIHelpers>(m, "MPI")
      .def("rank", &MPIHelpers::MPIRank,
           R"EOF(int: The MPI rank for the current process.  )EOF")
      .def("size", &MPIHelpers::MPISize,
           R"EOF(int: The total number of MPI ranks currently active.  )EOF");
}

}  // namespace netket

#endif
