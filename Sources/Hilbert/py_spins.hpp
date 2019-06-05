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

#ifndef NETKET_PYSPINS_HPP
#define NETKET_PYSPINS_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "spins.hpp"

namespace py = pybind11;

namespace netket {

void AddSpins(py::module &subm) {
  py::class_<Spin, AbstractHilbert>(
      subm, "Spin", R"EOF(Hilbert space composed of spin states.)EOF")
      .def(py::init<const AbstractGraph &, double>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("s"), R"EOF(
           Constructs a new ``Spin`` given a graph and the value of each spin.

           Args:
               graph: Graph representation of sites.
               s: Spin at each site. Must be integer or half-integer.

           Examples:
               Simple spin hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Spin
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Spin(graph=g, s=0.5)
               >>> print(hi.size)
               100

               ```
           )EOF")
      .def(py::init<const AbstractGraph &, double, double>(),
           py::keep_alive<1, 2>(), py::arg("graph"), py::arg("s"),
           py::arg("total_sz"), R"EOF(
           Constructs a new ``Spin`` given a graph and the value of each spin.

           Args:
               graph: Graph representation of sites.
               s: Spin at each site. Must be integer or half-integer.
               total_sz: Constrain total spin of system to a particular value.

           Examples:
               Simple spin hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Spin
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Spin(graph=g, s=0.5, total_sz=0)
               >>> print(hi.size)
               100

               ```
           )EOF");
}
}  // namespace netket
#endif
