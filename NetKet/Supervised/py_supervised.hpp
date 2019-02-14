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

#ifndef NETKET_PYSUPERVISED_HPP
#define NETKET_PYSUPERVISED_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "supervised.hpp"

namespace py = pybind11;

namespace netket {

void AddSupervisedModule(py::module &m) {
  auto subm = m.def_submodule("supervised");

  py::class_<Supervised>(subm, "supervised", R"EOF(Supervised learning scheme to learn data, i.e. the given state, by stochastic gradient descent with log overlap loss or MSE loss.)EOF")
      .def(py::init([](MachineType &ma, AbstractOptimizer &op, int batch_size,
                       std::vector<Eigen::VectorXd> samples,
                       std::vector<Eigen::VectorXcd> targets) {
             return Supervised{ma,
                               op,
                               batch_size,
                               std::move(samples),
                               std::move(targets)};
           }),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("machine"),
           py::arg("optimizer"), py::arg("batch_size"), py::arg("samples"),
           py::arg("targets"),
	   R"EOF(
           Construct a Supervised object given a machine, an optimizer, batch size and
           data, including samples and targets.

           Args:
               machine: The machine representing the wave function.
               optimizer: The optimizer object that determines how the SGD optimization.
               batch_size: The batch size used in SGD.
               samples: The input data, i.e. many-body basis.
               targets: The output label, i.e. amplitude of the corresponding basis.

           )EOF")
      .def_property_readonly("loss_log_overlap", &Supervised::GetLogOverlap,
           R"EOF(double: The current negative log fidelity.)EOF")
      .def_property_readonly("loss_mse", &Supervised::GetMse,
           R"EOF(double: The mean square error of amplitudes.)EOF")
      .def_property_readonly("loss_mse_log", &Supervised::GetMseLog,
           R"EOF(double: The mean square error of the log of amplitudes.)EOF")
      .def("run", &Supervised::Run, py::arg("n_iter"),
           py::arg("loss_function") = "Overlap_phi", py::arg("output_prefix") = "output",
	   py::arg("save_params_every") = 50, R"EOF(
           Run supervised learning.

           Args:
               n_iter: The number of iterations for running.
               loss_function: The loss function choosing for learning, Default: Overlap_phi
               output_prefix: The output file name, without extension.
               save_params_every: Frequency to dump wavefunction parameters. The default is 50.

           )EOF")
      .def("iterate", &Supervised::Iterate, py::arg("loss_function") = "Overlap_phi", R"EOF(
           Run one iteration of supervised learning. This should be helpful for testing and
           having self-defined control sequence in python.

           Args:
               loss_function: The loss function choosing for learning, Default: Overlap_phi

           )EOF");
}

}  // namespace netket

#endif
