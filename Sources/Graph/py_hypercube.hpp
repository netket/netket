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

#ifndef NETKET_PYHYPERCUBE_HPP
#define NETKET_PYHYPERCUBE_HPP

#include "hypercube.hpp"

namespace py = pybind11;

namespace netket {

void AddHypercube(py::module& subm) {
  py::class_<Hypercube, AbstractGraph>(subm, "Hypercube",
                                       R"EOF(
         A hypercube lattice of side L in d dimensions.
         Periodic boundary conditions can also be imposed.)EOF")
      .def(py::init<int, int, bool>(), py::arg("length"), py::arg("n_dim") = 1,
           py::arg("pbc") = true, R"EOF(
         Constructs a new ``Hypercube`` given its side length and dimension.

         Args:
             length: Side length of the hypercube.
                 It must always be >=1,
                 but if ``pbc==True`` then the minimal
                 valid length is 3.
             n_dim: Dimension of the hypercube. It must be at least 1.
             pbc: If ``True`` then the constructed hypercube
                 will have periodic boundary conditions, otherwise
                 open boundary conditions are imposed.

         Examples:
             A 10x10 square lattice with periodic boundary conditions can be
             constructed as follows:

             ```python
             >>> from netket.graph import Hypercube
             >>> g=Hypercube(length=10,n_dim=2,pbc=True)
             >>> print(g.n_sites)
             100

             ```
         )EOF")
      .def(py::init([](int length, py::iterable xs) {
             auto iterator = xs.attr("__iter__")();
             return Hypercube{length, detail::Iterable2ColorMap(iterator)};
           }),
           py::arg("length"), py::arg("colors"), R"EOF(
         Constructs a new `Hypercube` given its side length and edge coloring.

         Args:
             length: Side length of the hypercube.
                 It must always be >=3 if the
                 hypercube has periodic boundary conditions
                 and >=1 otherwise.
             colors: Edge colors, must be an iterable of
                 `Tuple[int, int, int]` where each
                 element `(i, j, c) represents an
                 edge `i <-> j` of color `c`.
                 Colors must be assigned to **all** edges.)EOF");
}
}  // namespace netket
#endif
