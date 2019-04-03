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

#ifndef NETKET_PYQUANTUM_STATE_RECONSTRUCTION_HPP
#define NETKET_PYQUANTUM_STATE_RECONSTRUCTION_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "quantum_state_reconstruction.hpp"

namespace py = pybind11;

namespace netket {

void AddUnsupervisedModule(py::module &m) {
  auto subm = m.def_submodule("unsupervised");

  py::class_<QuantumStateReconstruction>(subm, "Qsr")
      .def(
          py::init([](AbstractSampler &sa, AbstractOptimizer &op,
                      int batch_size, int n_samples, int niter_opt,
                      py::tuple rotations, std::vector<Eigen::VectorXd> samples,
                      std::vector<int> bases, std::string output_file,
                      int discarded_samples, int discarded_samples_on_init) {
            auto rots = py::cast<std::vector<AbstractOperator *>>(rotations);
            return QuantumStateReconstruction{sa,
                                              op,
                                              batch_size,
                                              n_samples,
                                              niter_opt,
                                              std::move(rots),
                                              std::move(samples),
                                              std::move(bases),
                                              output_file,
                                              discarded_samples,
                                              discarded_samples_on_init};
          }),
          py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
          py::keep_alive<1, 7>(), py::arg("sampler"), py::arg("optimizer"),
          py::arg("batch_size"), py::arg("n_samples"), py::arg("niter_opt"),
          py::arg("rotations"), py::arg("samples"), py::arg("bases"),
          py::arg("output_file"), py::arg("discarded_samples") = -1,
          py::arg("discarded_samples_on_init") = 0)
      .def("add_observable", &QuantumStateReconstruction::AddObservable,
           py::keep_alive<1, 2>())
      .def("run", &QuantumStateReconstruction::Run);
}

}  // namespace netket

#endif
