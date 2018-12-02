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

#ifndef NETKET_PYHILBERT_HPP
#define NETKET_PYHILBERT_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "hilbert.hpp"

namespace py = pybind11;

namespace netket {

constexpr int HilbertIndex::MaxStates;

#define ADDHILBERTMETHODS(name)                \
                                               \
  .def("is_discrete", &name::IsDiscrete)       \
      .def("local_size", &name::LocalSize)     \
      .def("size", &name::Size)                \
      .def("local_states", &name::LocalStates) \
      .def("random_vals", &name ::RandomVals)  \
      .def("update_conf", &name::UpdateConf)

void AddHilbertModule(py::module &m) {
  auto subm = m.def_submodule("hilbert");

  py::class_<Hilbert>(subm, "Hilbert")
      .def(py::init<Spin>())
      .def(py::init<Qubit>())
      .def(py::init<Boson>())
      .def(py::init<CustomHilbert>()) ADDHILBERTMETHODS(Hilbert);

  py::class_<Spin>(subm, "Spin")
      .def(py::init<Graph, double>(), py::arg("graph"), py::arg("s"))
      .def(py::init<Graph, double, double>(), py::arg("graph"), py::arg("s"),
           py::arg("total_sz")) ADDHILBERTMETHODS(Spin);
  py::implicitly_convertible<Spin, Hilbert>();

  py::class_<Qubit>(subm, "Qubit")
      .def(py::init<Graph>(), py::arg("graph")) ADDHILBERTMETHODS(Qubit);
  py::implicitly_convertible<Qubit, Hilbert>();

  py::class_<Boson>(subm, "Boson")
      .def(py::init<Graph, int>(), py::arg("graph"), py::arg("n_max"))
      .def(py::init<Graph, int, int>(), py::arg("graph"), py::arg("n_max"),
           py::arg("n_bosons")) ADDHILBERTMETHODS(Boson);
  py::implicitly_convertible<Boson, Hilbert>();

  py::class_<CustomHilbert>(subm, "CustomHilbert")
      .def(py::init<Graph, std::vector<double>>(), py::arg("graph"),
           py::arg("local_states")) ADDHILBERTMETHODS(CustomHilbert);
  py::implicitly_convertible<CustomHilbert, Hilbert>();

  py::class_<HilbertIndex>(subm, "HilbertIndex")
      .def(py::init<const Hilbert &>(), py::arg("hilbert"))
      .def("n_states", &HilbertIndex::NStates)
      .def("number_to_state", &HilbertIndex::NumberToState)
      .def("state_to_number", &HilbertIndex::StateToNumber)
      .def_readonly_static("max_states", &HilbertIndex::MaxStates);

}  // namespace netket

}  // namespace netket

#endif
