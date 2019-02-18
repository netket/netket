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

#ifndef NETKET_PYBOSONS_HPP
#define NETKET_PYBOSONS_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "bosons.hpp"

namespace py = pybind11;

namespace netket {

void AddBosons(py::module &subm) {
  py::class_<Boson, AbstractHilbert>(
      subm, "Boson", R"EOF(Hilbert space composed of bosonic states.)EOF")
      .def(py::init<const AbstractGraph &, int>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("n_max"), R"EOF(
           Constructs a new ``Boson`` given a graph and maximum occupation number.

           Args:
               graph: Graph representation of sites.
               n_max: Maximum occupation for a site.

           Examples:
               Simple boson hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Boson
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Boson(graph=g, n_max=4)
               >>> print(hi.size)
               100

               ```
           )EOF")
      .def(py::init<const AbstractGraph &, int, int>(), py::keep_alive<1, 2>(),
           py::arg("graph"), py::arg("n_max"), py::arg("n_bosons"), R"EOF(
           Constructs a new ``Boson`` given a graph,  maximum occupation number,
           and total number of bosons.

           Args:
               graph: Graph representation of sites.
               n_max: Maximum occupation for a site.
               n_bosons: Constraint for the number of bosons.

           Examples:
               Simple boson hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import Boson
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = Boson(graph=g, n_max=5, n_bosons=11)
               >>> print(hi.size)
               100

               ```
           )EOF");
}

}  // namespace netket

#endif
