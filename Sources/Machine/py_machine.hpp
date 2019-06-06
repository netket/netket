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

#ifndef NETKET_PYMACHINE_HPP
#define NETKET_PYMACHINE_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <limits>
#include <vector>
#include "Layers/py_layer.hpp"
#include "abstract_machine.hpp"
#include "py_ffnn.hpp"
#include "py_jastrow.hpp"
#include "py_jastrow_symm.hpp"
#include "py_mps_periodic.hpp"
#include "py_rbm_multival.hpp"
#include "py_rbm_spin.hpp"
#include "py_rbm_spin_phase.hpp"
#include "py_rbm_spin_real.hpp"
#include "py_rbm_spin_symm.hpp"
#include "DensityMatrices/py_density_matrix.hpp"

namespace py = pybind11;

namespace netket {

void AddMachineModule(py::module &m) {
  auto subm = m.def_submodule("machine");

  py::class_<AbstractMachine>(subm, "Machine")
      .def_property_readonly(
          "n_par", &AbstractMachine::Npar,
          R"EOF(int: The number of parameters in the machine.)EOF")
      .def_property("parameters", &AbstractMachine::GetParameters,
                    &AbstractMachine::SetParameters,
                    R"EOF(list: List containing the parameters within the layer.
            Read and write)EOF")
      .def("init_random_parameters", &AbstractMachine::InitRandomPars,
           py::arg("seed") = 1234, py::arg("sigma") = 0.1,
           R"EOF(
             Member function to initialise machine parameters.

             Args:
                 seed: The random number generator seed.
                 sigma: Standard deviation of normal distribution from which
                     parameters are drawn.
           )EOF")
      .def("log_val",
           (Complex(AbstractMachine::*)(AbstractMachine::VisibleConstType)) &
               AbstractMachine::LogVal,
           py::arg("v"),
           R"EOF(
                 Member function to obtain log value of machine given an input
                 vector.

                 Args:
                     v: Input vector to machine.
           )EOF")
      .def("log_val_diff",
           (AbstractMachine::VectorType(AbstractMachine::*)(
               AbstractMachine::VisibleConstType,
               const std::vector<std::vector<int>> &,
               const std::vector<std::vector<double>> &)) &
               AbstractMachine::LogValDiff,
           py::arg("v"), py::arg("tochange"), py::arg("newconf"),
           R"EOF(
                 Member function to obtain difference in log value of machine
                 given an input and a change to the input.

                 Args:
                     v: Input vector to machine.
                     tochange: list containing the indices of the input to be
                         changed
                     newconf: list containing the new (changed) values at the
                         indices specified in tochange
           )EOF")
      .def("der_log",
           (AbstractMachine::VectorType(AbstractMachine::*)(
               AbstractMachine::VisibleConstType)) &
               AbstractMachine::DerLog,
           py::arg("v"),
           R"EOF(
                 Member function to obtain the derivatives of log value of
                 machine given an input wrt the machine's parameters.

                 Args:
                     v: Input vector to machine.
           )EOF")
      .def_property_readonly(
          "n_visible", &AbstractMachine::Nvisible,
          R"EOF(int: The number of inputs into the machine aka visible units in
            the case of Restricted Boltzmann Machines.)EOF")
      .def_property_readonly(
          "hilbert", &AbstractMachine::GetHilbert,
          R"EOF(netket.hilbert.Hilbert: The hilbert space object of the system.)EOF")
      .def_property_readonly(
          "is_holomorphic", &AbstractMachine::IsHolomorphic,
          R"EOF(bool: Whether the given wave-function is a holomorphic function of
            its parameters )EOF")
      .def("save",
           [](const AbstractMachine &a, std::string filename) {
             json j;
             a.to_json(j);
             std::ofstream filewf(filename);
             filewf << j << std::endl;
             filewf.close();
           },
           py::arg("filename"),
           R"EOF(
                 Member function to save the machine parameters.

                 Args:
                     filename: name of file to save parameters to.
           )EOF")
      .def("load",
           [](AbstractMachine &a, std::string filename) {
             std::ifstream filewf(filename);
             if (filewf.is_open()) {
               json j;
               filewf >> j;
               filewf.close();
               a.from_json(j);
             }
           },
           py::arg("filename"),
           R"EOF(
                 Member function to load machine parameters from a json file.

                 Args:
                     filename: name of file to load parameters from.
           )EOF")
      .def("to_array",
           [](AbstractMachine &self) -> AbstractMachine::VectorType {
             const auto &hind = self.GetHilbert().GetIndex();
             AbstractMachine::VectorType vals(hind.NStates());

             double maxlog = std::numeric_limits<double>::lowest();

             for (Index i = 0; i < hind.NStates(); i++) {
               vals(i) = self.LogVal(hind.NumberToState(i));
               if (std::real(vals(i)) > maxlog) {
                 maxlog = std::real(vals(i));
               }
             }

             for (Index i = 0; i < hind.NStates(); i++) {
               vals(i) -= maxlog;
               vals(i) = std::exp(vals(i));
             }

             vals /= vals.norm();
             return vals;
           },
           R"EOF(
                Returns a numpy array representation of the machine.
                The returned array is normalized to 1 in L2 norm.
                Note that, in general, the size of the array is exponential
                in the number of quantum numbers, and this operation should thus
                only be performed for low-dimensional Hilbert spaces.

                This method requires an indexable Hilbert space.
              )EOF")
      .def("log_norm",
           [](AbstractMachine &self) -> double {
             const auto &hind = self.GetHilbert().GetIndex();
             AbstractMachine::VectorType vals(hind.NStates());

             double maxlog = std::numeric_limits<double>::lowest();

             for (Index i = 0; i < hind.NStates(); i++) {
               vals(i) = self.LogVal(hind.NumberToState(i));
               if (std::real(vals(i)) > maxlog) {
                 maxlog = std::real(vals(i));
               }
             }

             double norpsi = 0;
             for (Index i = 0; i < hind.NStates(); i++) {
               vals(i) -= maxlog;
               norpsi += std::norm(std::exp(vals(i)));
             }

             return std::log(norpsi) + 2. * maxlog;
           },
           R"EOF(
                Returns the log of the L2 norm of the wave-function.
                This operation is a brute-force calculation, and should thus
                only be performed for low-dimensional Hilbert spaces.

                This method requires an indexable Hilbert space.
                )EOF");

  AddRbmSpin(subm);
  AddRbmSpinSymm(subm);
  AddRbmMultival(subm);
  AddRbmSpinReal(subm);
  AddRbmSpinPhase(subm);
  AddJastrow(subm);
  AddJastrowSymm(subm);
  AddMpsPeriodic(subm);
  AddFFNN(subm);
  AddLayerModule(m);

  AddDensityMatrixModule(subm);
}

}  // namespace netket

#endif
