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

#ifndef NETKET_PYQUBITS_HPP
#define NETKET_PYQUBITS_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "qubits.hpp"

namespace py = pybind11;

namespace netket {

void AddQubits(py::module &subm) {
  py::class_<Qubit, AbstractHilbert>(
      subm, "Qubit", R"EOF(Hilbert space composed of qubits.)EOF")
      .def(py::init<const AbstractGraph &>(), py::keep_alive<1, 2>(),
           py::arg("graph"), R"EOF(
           Constructs a new ``Qubit`` given a graph.

           Args:
               graph: Graph representation of sites.

           Examples:
               Simple qubit hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Qubit
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Qubit(graph=g)
               >>> print(hi.size)
               100

               ```
           )EOF")
      .def(py::init<int>(), py::arg("n") = 1, R"EOF(
           Constructs a new Hilbert space tensor product of n ``Qubit`` spaces .

           Args:
               n: Total number of qubits.

           Examples:
               Simple qubit hilbert space.

               ```python
               >>> from netket.hilbert import Qubit
               >>> hi = Qubit(10)
               >>> print(hi.size)
               10

           ```
         )EOF");
}

}  // namespace netket

#endif
