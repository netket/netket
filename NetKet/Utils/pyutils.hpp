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
#include <memory>
#include <vector>
#include "all_utils.hpp"

namespace py = pybind11;

namespace netket {

struct PyMPIEnvironment {
  PyMPIEnvironment() {
    int already_initialized;
    MPI_Initialized(&already_initialized);
    if (!already_initialized) {
      // We don't have access to command-line arguments
      if (MPI_Init(nullptr, nullptr) != MPI_SUCCESS) {
        std::ostringstream msg;
        msg << "This should never have happened. How did you manage to "
               "call MPI_Init() in between two C function calls?! "
               "Terminating now.";
        std::cerr << msg.str() << std::endl;
        std::terminate();
      }
      have_initialized_ = true;
#if !defined(NDEBUG)
      std::cerr << "MPI successfully initialized by NetKet." << std::endl;
#endif
    }
  }
  ~PyMPIEnvironment() {
    if (have_initialized_) {
      // We have initialized MPI so it's only right we finalize it.
      MPI_Finalize();
#if !defined(NDEBUG)
      std::cerr << "MPI successfully finalized by NetKet." << std::endl;
#endif
    }
  }

 private:
  bool have_initialized_;
};

void AddUtilsModule(py::module &m) {
  // The MPI environment is added to the main module
  py::class_<PyMPIEnvironment, std::shared_ptr<PyMPIEnvironment>>(
      m, "mpi_environment");
  m.attr("_mpi_environment") = py::cast(std::make_shared<PyMPIEnvironment>());

  auto subm = m.def_submodule("utils");

  py::class_<netket::default_random_engine>(subm, "RandomEngine")
      .def(py::init<netket::default_random_engine::result_type>(),
           py::arg("seed") = netket::default_random_engine::default_seed)
      .def("Seed", (void (netket::default_random_engine::*)(
                       netket::default_random_engine::result_type)) &
                       netket::default_random_engine::seed);

  py::class_<Lookup<double>>(m, "LookupReal").def(py::init<>());

  py::class_<Lookup<std::complex<double>>>(m, "LookupComplex")
      .def(py::init<>());
}

}  // namespace netket

#endif
