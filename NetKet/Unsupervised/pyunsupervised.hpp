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
  using MatType = LocalOperator::MatType;
  py::class_<QuantumStateReconstruction>(subm, "Qsr")
      .def(py::init<SamplerType &,AbstractOptimizer &,int,int,int,std::vector<MatType>,std::vector<std::vector<int> >,std::vector<Eigen::VectorXd>, std::vector<int>,std::string,int, int>(),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(),
           py::arg("sampler"),
           py::arg("optimizer"),py::arg("batch_size"),py::arg("n_samples"),py::arg("niter_opt"), py::arg("rotations"),py::arg("sites"),py::arg("samples"),py::arg("bases"), 
           py::arg("output_file"),
           py::arg("discarded_samples") = -1,
           py::arg("discarded_samples_on_init") = 0)
           .def("run", &QuantumStateReconstruction::Run).def("add_observable",&QuantumStateReconstruction::AddObservable);
}

}  // namespace netket

#endif
