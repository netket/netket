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

#ifndef NETKET_PYCUSTOMHILBERT_HPP
#define NETKET_PYCUSTOMHILBERT_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "custom_hilbert.hpp"

namespace py = pybind11;

namespace netket {

void AddCustomHilbert(py::module &subm) {
  py::class_<CustomHilbert, AbstractHilbert>(subm, "CustomHilbert",
                                             R"EOF(A custom hilbert space.)EOF")
      .def(py::init<const AbstractGraph &, std::vector<double>>(),
           py::keep_alive<1, 2>(), py::arg("graph"), py::arg("local_states"),
           R"EOF(
           Constructs a new ``CustomHilbert`` given a graph and a list of
           eigenvalues of the states.

           Args:
               graph: Graph representation of sites.
               local_states: Eigenvalues of the states.

           Examples:
               Simple custom hilbert space.

               ```python
               >>> from netket.graph import Hypercube
               >>> from netket.hilbert import CustomHilbert
               >>> g = Hypercube(length=10,n_dim=2,pbc=True)
               >>> hi = CustomHilbert(graph=g, local_states=[-1232, 132, 0])
               >>> print(hi.size)
               100

               ```
           )EOF")
      .def(py::init<int, std::vector<double>>(), py::arg("n") = 1,
           py::arg("local_states"),
           R"EOF(
           Constructs a new ``CustomHilbert`` given a number of sites and a list of
           local quantum numbers.

           Args:
               n: Number of sites.
               local_states: Eigenvalues of the states.

           Examples:
               Simple custom hilbert space.

               ```python
               >>> from netket.hilbert import CustomHilbert
               >>> hi = CustomHilbert(n=100, local_states=[-1232, 132, 0])
               >>> print(hi.size)
               100

               ```
           )EOF");
}

}  // namespace netket

#endif
