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

#ifndef NETKET_PYGROUND_STATE_HPP
#define NETKET_PYGROUND_STATE_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ground_state.hpp"

namespace py = pybind11;

namespace netket {

void AddGroundStateModule(py::module &m) {
  auto subm = m.def_submodule("gs");

  py::class_<VariationalMonteCarlo>(subm, "Vmc")
      .def(py::init<AbstractOperator &, AbSamplerType &, AbstractOptimizer &,
                    int, int, std::string, int, int, std::string, double, bool,
                    bool, bool, int>(),
           py::arg("hamiltonian"), py::arg("sampler"), py::arg("optimizer"),
           py::arg("nsamples"), py::arg("niter_opt"), py::arg("output_file"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0, py::arg("method") = "Sr",
           py::arg("diag_shift") = 0.01, py::arg("rescale_shift") = false,
           py::arg("use_iterative") = false, py::arg("use_cholesky") = true,
           py::arg("save_every") = 50)
      .def("AddObservable", &VariationalMonteCarlo::AddObservable)
      .def("Run", &VariationalMonteCarlo::Run);

  py::class_<ImaginaryTimeDriver>(subm, "ImaginaryTimeDriver")
      .def(py::init<ImaginaryTimeDriver::Matrix &,
                    ImaginaryTimeDriver::Stepper &, JsonOutputWriter &, double,
                    double, double>(),
           py::arg("hamiltonian"), py::arg("stepper"), py::arg("output_writer"),
           py::arg("tmin"), py::arg("tmax"), py::arg("dt"))
      .def("add_observable", &ImaginaryTimeDriver::AddObservable,
           py::arg("observable"), py::arg("name"),
           py::arg("matrix_type") = "Sparse")
      .def("run", &ImaginaryTimeDriver::Run, py::arg("initial_state"));

  py::class_<eddetail::result_t>(subm, "EdResult")
      .def_readwrite("eigenvalues", &eddetail::result_t::eigenvalues)
      .def_readwrite("eigenvectors", &eddetail::result_t::eigenvectors)
      .def_readwrite("which_eigenvector",
                     &eddetail::result_t::which_eigenvector);

  subm.def("LanczosEd", &lanczos_ed, py::arg("operator"),
           py::arg("matrix_free") = false, py::arg("first_n") = 1,
           py::arg("max_iter") = 1000, py::arg("seed") = 42,
           py::arg("precision") = 1.0e-14, py::arg("get_groundstate") = false);
}

}  // namespace netket

#endif
