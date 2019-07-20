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

#ifndef NETKET_PYNQS_HPP
#define NETKET_PYNQS_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "nqs.hpp"

namespace py = pybind11;

namespace netket {

void AddNQSModule(py::module &m) {
  auto subm = m.def_submodule("nqs");

  py::class_<NQS>(
      subm, "NQS",
      R"EOF(Neural Quantum state for classical simulation of quantum circuits using RBMs.)EOF")
      .def(py::init([](int nqubits) {
             return NQS{nqubits};
           }),
           py::arg("nqubits"),
           R"EOF(
           Construct a NQS object with the given number of qubits.

           Args:
               nqubits: Number of Qubits of the circuit to be simulated.
           )EOF")
      .def("applyHadamard", &NQS::applyHadamard, py::arg("qubit"),
            R"EOF(
           Apply Hadamard gate to qubit.

           Args:
               qubit: The index of the qubit the gate will be applied to
           )EOF")
      .def("applyPauliX", &NQS::applyPauliX, py::arg("qubit"),
            R"EOF(
           Apply Pauli X gate to qubit.

           Args:
               qubit: The index of the qubit the gate will be applied to
           )EOF")
      .def("applyPauliY", &NQS::applyPauliY, py::arg("qubit"),
            R"EOF(
           Apply Pauli Y gate to qubit.

           Args:
               qubit: The index of the qubit the gate will be applied to
           )EOF")
      .def("applyPauliZ", &NQS::applyPauliZ, py::arg("qubit"),
            R"EOF(
           Apply Pauli Z gate to qubit.

           Args:
               qubit: The index of the qubit the gate will be applied to
           )EOF")
      .def("applySingleZRotation", &NQS::applySingleZRotation,
           py::arg("qubit"), py::arg("theta"), R"EOF(
           Apply Z rotation to qubit.

           Args:
               qubit: The index of the qubit the gate will be applied to
               theta: angle of the rotation

           )EOF")
      .def("applyControlledZRotation", &NQS::applyControlledZRotation, py::arg("controlQubit"),
            py::arg("qubit"), py::arg("theta"),
            R"EOF(
           Apply Controlled Z rotation to qubit.

           Args:
               qubit: The index of the qubit the gate will be applied to
               controlQubit: The index of the qubit depending on which value the rotation will be applied
               theta: angle of the rotation
           )EOF")
      .def("sample", &NQS::sample,
            R"EOF(
           Sample from the nqs.
           )EOF")
      .def("getPsiParams", &NQS::getPsiParams,
            R"EOF(
           Get parameters of the underlying Boltzmann machine.
           )EOF");
}

}  // namespace netket

#endif
