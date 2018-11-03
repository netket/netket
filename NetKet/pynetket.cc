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

#ifndef NETKET_PYNETKET_CC
#define NETKET_PYNETKET_CC

#include <mpi.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "netket.hpp"

namespace py = pybind11;

namespace netket {

PYBIND11_MODULE(pynetket, m) {
  py::class_<Graph>(m, "Graph")
      .def(py::init<std::string, py::kwargs>())
      .def("Nsites", &Graph::Nsites)
      .def("AdjacencyList", &Graph::AdjacencyList)
      .def("SymmetryTable", &Graph::SymmetryTable)
      .def("EdgeColors", &Graph::EdgeColors)
      .def("IsBipartite", &Graph::IsBipartite)
      .def("IsConnected", &Graph::IsConnected)
      .def("Distances", &Graph::Distances)
      .def("AllDistances", &Graph::AllDistances);

  py::class_<Hilbert>(m, "Hilbert")
      .def(py::init<py::kwargs>())
      .def(py::init<const Graph&, py::kwargs>())
      .def("IsDiscrete", &Hilbert::IsDiscrete)
      .def("LocalSize", &Hilbert::LocalSize)
      .def("Size", &Hilbert::Size)
      .def("LocalStates", &Hilbert::LocalStates)
      .def("UpdateConf", &Hilbert::UpdateConf);
}

}  // namespace netket

#endif
