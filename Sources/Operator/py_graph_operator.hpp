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

#ifndef NETKET_PYGRAPHOPERATOR_HPP
#define NETKET_PYGRAPHOPERATOR_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "graph_operator.hpp"

namespace py = pybind11;

namespace netket {

void AddGraphOperator(py::module &subm) {
  py::class_<GraphOperator, AbstractOperator>(
      subm, "GraphOperator", R"EOF(A custom graph based operator.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>,
                    GraphOperator::OVecType, GraphOperator::OVecType,
                    std::vector<int>>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"),
           py::arg("siteops") = GraphOperator::OVecType(),
           py::arg("bondops") = GraphOperator::OVecType(),
           py::arg("bondops_colors") = std::vector<int>(), R"EOF(
         Constructs a new ``GraphOperator`` given a hilbert space and either a
         list of operators acting on sites or a list acting on the bonds.
         Users can specify the color of the bond that an operator acts on, if
         desired. If none are specified, the bond operators act on all edges.

         Args:
             hilbert: Hilbert space the operator acts on.
             siteops: A list of operators that act on the nodes of the graph.
                 The default is an empty list. Note that if no siteops are
                 specified, the user must give a list of bond operators.
             bondops: A list of operators that act on the edges of the graph.
                 The default is an empty list. Note that if no bondops are
                 specified, the user must give a list of site operators.
             bondops_colors: A list of edge colors, specifying the color each
                 bond operator acts on. The defualt is an empty list.

         Examples:
             Constructs a ``BosGraphOperator`` operator for a 2D system.

             ```python
             >>> import netket as nk
             >>> sigmax = [[0, 1], [1, 0]]
             >>> mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
             >>> edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
             ... [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
             ... [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]
             >>> g = nk.graph.CustomGraph(edges=edges)
             >>> hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], graph=g)
             >>> op = nk.operator.GraphOperator(
             ... hi, siteops=[sigmax], bondops=[mszsz], bondops_colors=[0])
             >>> print(op.hilbert.size)
             20

             ```
         )EOF")
      .def(py::self + py::self);
}

}  // namespace netket

#endif
