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
      .def(py::init<const AbstractOperator &, SamplerType &,
                    AbstractOptimizer &, int, int, int, const std::string &,
                    double, bool, bool, bool>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::keep_alive<1, 4>(), py::arg("hamiltonian"), py::arg("sampler"),
           py::arg("optimizer"), py::arg("n_samples"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0, py::arg("method") = "Sr",
           py::arg("diag_shift") = 0.01, py::arg("rescale_shift") = false,
           py::arg("use_iterative") = false, py::arg("use_cholesky") = true)
      .def_property_readonly("psi", &VariationalMonteCarlo::GetPsi)
      .def("add_observable", &VariationalMonteCarlo::AddObservable,
           py::keep_alive<1, 2>())
      .def("run", &VariationalMonteCarlo::Run, py::arg("filename_prefix"),
           py::arg("max_iter"), py::arg("step_size") = 1,
           py::arg("save_params_every") = 50)
      .def("iter", &VariationalMonteCarlo::Iterate, py::arg("max_iter"),
           py::arg("step_size") = 1, py::arg("store_params") = true);

  py::class_<VmcState>(subm, "VmcState")
      .def_readonly("current_step", &VmcState::current_step)
      .def_readonly("acceptance", &VmcState::acceptance)
      .def_readonly("observables", &VmcState::observables)
      .def_readonly("params", &VmcState::parameters)
      .def("__repr__", [](const VmcState &self) {
        std::stringstream str;
        str << "<VmcState: step=" << self.current_step << ">";
        return str.str();
      });

  py::class_<VmcIterator>(subm, "VmcIterator")
      .def("__iter__", [](VmcIterator &self) {
        return py::make_iterator(self.begin(), self.end());
      });

  py::class_<ImaginaryTimeDriver>(subm, "ImaginaryTimeDriver")
      .def(py::init<ImaginaryTimeDriver::Matrix &,
                    ImaginaryTimeDriver::Stepper &, JsonOutputWriter &, double,
                    double, double>(),
           py::arg("hamiltonian"), py::arg("stepper"), py::arg("output_writer"),
           py::arg("tmin"), py::arg("tmax"), py::arg("dt"))
      .def("add_observable", &ImaginaryTimeDriver::AddObservable,
           py::keep_alive<1, 2>(), py::arg("observable"), py::arg("name"),
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
