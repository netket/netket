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
  auto vmc = subm.def_submodule("vmc");

  py::class_<VariationalMonteCarlo>(vmc, "Vmc")
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

  py::class_<VariationalMonteCarlo::Step>(vmc, "Step")
      .def_readonly("current_step", &VariationalMonteCarlo::Step::index)
      .def_readonly("acceptance", &VariationalMonteCarlo::Step::acceptance)
      .def_readonly("observables", &VariationalMonteCarlo::Step::observables)
      .def_readonly("params", &VariationalMonteCarlo::Step::parameters)
      .def("__repr__", [](const VariationalMonteCarlo::Step &self) {
        std::stringstream str;
        str << "<VmcStep: step=" << self.index << ">";
        return str.str();
      });

  py::class_<VariationalMonteCarlo::Iterator>(vmc, "Iterator")
      .def("__iter__", [](VariationalMonteCarlo::Iterator &self) {
        return py::make_iterator(self.begin(), self.end());
      });

  auto excact = subm.def_submodule("exact");

  py::class_<ImagTimePropagation>(excact, "ImagTimePropagation")
      .def(py::init<ImagTimePropagation::Matrix &,
                    ImagTimePropagation::Stepper &, double,
                    ImagTimePropagation::StateVector>(),
           py::arg("hamiltonian"), py::arg("stepper"), py::arg("t0"),
           py::arg("initial_state"))
      .def("add_observable", &ImagTimePropagation::AddObservable,
           py::keep_alive<1, 2>(), py::arg("observable"), py::arg("name"),
           py::arg("matrix_type") = "Sparse")
      .def("iter", &ImagTimePropagation::Iterate, py::arg("dt"),
           py::arg("max_steps"),
           py::arg("store_state") = true)
      .def_property("t", &ImagTimePropagation::GetTime,
                    &ImagTimePropagation::SetTime);

  py::class_<ImagTimePropagation::Step>(excact, "ImagTimeStep")
      .def_readonly("current_step", &ImagTimePropagation::Step::index)
      .def_readonly("t", &ImagTimePropagation::Step::t)
      .def_readonly("observables", &ImagTimePropagation::Step::observables)
      .def_readonly("state", &ImagTimePropagation::Step::state)
      .def("__repr__", [](const ImagTimePropagation::Step &self) {
        std::stringstream str;
        str << "<ImagTimeStep: step=" << self.index << ", it=" << self.t << ">";
        return str.str();
      });

  py::class_<ImagTimePropagation::Iterator>(excact, "ImagTimeIterator")
      .def("__iter__", [](ImagTimePropagation::Iterator &self) {
        return py::make_iterator(self.begin(), self.end());
      });

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
