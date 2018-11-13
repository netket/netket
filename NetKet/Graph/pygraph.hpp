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

#ifndef NETKET_PYGRAPH_HPP
#define NETKET_PYGRAPH_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "netket.hpp"

namespace py = pybind11;

namespace netket {

#define ADDGRAPHMETHODS(name)                              \
                                                           \
  .def("Nsites", &AbstractGraph::Nsites)                   \
  .def("AdjacencyList", &AbstractGraph::AdjacencyList)		       \
  .def("SymmetryTable", &AbstractGraph::SymmetryTable)		       \
  .def("EdgeColors", &AbstractGraph::EdgeColors)		       \
  .def("IsBipartite", &AbstractGraph::IsBipartite)		       \
  .def("IsConnected", &AbstractGraph::IsConnected)		       \
  .def("Distances", &AbstractGraph::Distances);

void AddGraphModule(py::module &m) {
  auto subm = m.def_submodule("graph");
  
  py::class_<AbstractGraph>(subm, "Graph") ADDGRAPHMETHODS(AbstractGraph);
  
  py::class_<Hypercube, AbstractGraph>(subm, "Hypercube")
    .def(py::init<int, int, bool, std::vector<std::vector<int>>>(),
	 py::arg("L"), py::arg("ndim"), py::arg("pbc") = true,
	 py::arg("edgecolors") = std::vector<std::vector<int>>())
    .def("Nsites", &Hypercube::Nsites)
    .def("AdjacencyList", &Hypercube::AdjacencyList)
    .def("SymmetryTable", &Hypercube::SymmetryTable)
    .def("EdgeColors", &Hypercube::EdgeColors)
    .def("IsBipartite", &Hypercube::IsBipartite)
    .def("IsConnected", &Hypercube::IsConnected)
    .def("Distances", &Hypercube::Distances)
    .def("AllDistances", &Hypercube::AllDistances) ADDGRAPHMETHODS(Hypercube);
  
  py::class_<CustomGraph, AbstractGraph>(subm, "CustomGraph")
    .def(
	 py::init<int, std::vector<std::vector<int>>,
	 std::vector<std::vector<int>>, std::vector<std::vector<int>>,
	 std::vector<std::vector<int>>, bool>(),
	 py::arg("size") = 0,
	 py::arg("adjacency_list") = std::vector<std::vector<int>>(),
	 py::arg("edges") = std::vector<std::vector<int>>(),
	 py::arg("automorphisms") = std::vector<std::vector<int>>(),
	 py::arg("edgecolors") = std::vector<std::vector<int>>(),
	 py::arg("is_bipartite") = false)
    .def("Nsites", &CustomGraph::Nsites)
    .def("AdjacencyList", &CustomGraph::AdjacencyList)
    .def("SymmetryTable", &CustomGraph::SymmetryTable)
    .def("EdgeColors", &CustomGraph::EdgeColors)
    .def("IsBipartite", &CustomGraph::IsBipartite)
    .def("IsConnected", &CustomGraph::IsConnected)
    .def("Distances", &CustomGraph::Distances)
    .def("AllDistances", &CustomGraph::AllDistances)
    ADDGRAPHMETHODS(CustomGraph);

}  // namespace netket

}  // namespace netket

#endif
