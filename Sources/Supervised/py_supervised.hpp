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

  py::class_<Supervised>(
      subm, "Supervised",
      R"EOF(Supervised learning scheme to learn data, i.e. the given state, by stochastic gradient descent with log overlap loss or MSE loss.)EOF")
      .def(py::init([](AbstractMachine &ma, AbstractOptimizer &op,
                       int batch_size, std::vector<Eigen::VectorXd> samples,
                       std::vector<Eigen::VectorXcd> targets,
                       const std::string &method, double diag_shift,
                       bool use_iterative, bool use_cholesky) {
             return Supervised{ma,
                               op,
                               batch_size,
                               std::move(samples),
                               std::move(targets),
                               method,
                               diag_shift,
                               use_iterative,
                               use_cholesky};
           }),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("machine"),
           py::arg("optimizer"), py::arg("batch_size"), py::arg("samples"),
           py::arg("targets"), py::arg("method") = "Gd",
           py::arg("diag_shift") = 0.01, py::arg("use_iterative") = false,
           py::arg("use_cholesky") = true,
           R"EOF(
           Construct a Supervised object given a machine, an optimizer, batch size and
           data, including samples and targets.

           Args:
               machine: The machine representing the wave function.
               optimizer: The optimizer object that determines how the SGD optimization.
               batch_size: The batch size used in SGD.
               samples: The input data, i.e. many-body basis.
               targets: The output label, i.e. amplitude of the corresponding basis.
               method: The chosen method to learn the parameters of the
                   wave-function. Possible choices are `Gd` (Regular Gradient descent),
                   and `Sr` (Stochastic reconfiguration a.k.a. natural gradient).
               diag_shift: The regularization parameter in stochastic
                   reconfiguration. The default is 0.01.
               use_iterative: Whether to use the iterative solver in the Sr
                   method (this is extremely useful when the number of
                   parameters to optimize is very large). The default is false.
               use_cholesky: Whether to use cholesky decomposition. The default
                   is true.

           )EOF")
      .def_property_readonly(
          "loss_log_overlap", &Supervised::GetLogOverlap,
          R"EOF(double: The current negative log fidelity.)EOF")
      .def_property_readonly(
          "loss_mse", &Supervised::GetMse,
          R"EOF(double: The mean square error of amplitudes.)EOF")
      .def_property_readonly(
          "loss_mse_log", &Supervised::GetMseLog,
          R"EOF(double: The mean square error of the log of amplitudes.)EOF")
      .def("run", &Supervised::Run, py::arg("n_iter"),
           py::arg("loss_function") = "Overlap_phi",
           py::arg("output_prefix") = "output",
           py::arg("save_params_every") = 50, R"EOF(
           Run supervised learning.

           Args:
               n_iter: The number of iterations for running.
               loss_function: The loss function choosing for learning, Default: Overlap_phi
               output_prefix: The output file name, without extension.
               save_params_every: Frequency to dump wavefunction parameters. The default is 50.

           )EOF")
      .def("advance", &Supervised::Advance,
           py::arg("loss_function") = "Overlap_phi", R"EOF(
           Run one iteration of supervised learning. This should be helpful for testing and
           having self-defined control sequence in python.

           Args:
               loss_function: The loss function choosing for learning, Default: Overlap_phi

           )EOF");
}

}  // namespace netket

#endif
