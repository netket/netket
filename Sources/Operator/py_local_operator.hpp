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

#ifndef NETKET_PYLOCALOPERATOR_HPP
#define NETKET_PYLOCALOPERATOR_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "local_operator.hpp"

namespace py = pybind11;

namespace netket {

void AddLocalOperator(py::module &subm) {
  py::class_<LocalOperator, AbstractOperator>(
      subm, "LocalOperator", R"EOF(A custom local operator.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, double>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("constant") = 0.,
           R"EOF(
           Constructs a new ``LocalOperator`` given a hilbert space and (if
           specified) a constant level shift.

           Args:
               hilbert: Hilbert space the operator acts on.
               constant: Level shift for operator. Default is 0.0.

           Examples:
               Constructs a ``LocalOperator`` without any operators.

               ```python
               >>> from netket.graph import CustomGraph
               >>> from netket.hilbert import CustomHilbert
               >>> from netket.operator import LocalOperator
               >>> g = CustomGraph(edges=[[i, i + 1] for i in range(20)])
               >>> hi = CustomHilbert(local_states=[1, -1], graph=g)
               >>> empty_hat = LocalOperator(hi)
               >>> print(len(empty_hat.acting_on))
               0

               ```
           )EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>,
                    std::vector<LocalOperator::MatType>,
                    std::vector<LocalOperator::SiteType>, double>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("operators"),
           py::arg("acting_on"), py::arg("constant") = 0., R"EOF(
          Constructs a new ``LocalOperator`` given a hilbert space, a vector of
          operators, a vector of sites, and (if specified) a constant level
          shift.

          Args:
              hilbert: Hilbert space the operator acts on.
              operators: A list of operators, in matrix form.
              acting_on: A list of sites, which the corresponding operators act
                  on.
              constant: Level shift for operator. Default is 0.0.

          Examples:
              Constructs a ``LocalOperator`` from a list of operators acting on
              a corresponding list of sites.

              ```python
              >>> from netket.graph import CustomGraph
              >>> from netket.hilbert import CustomHilbert
              >>> from netket.operator import LocalOperator
              >>> sx = [[0, 1], [1, 0]]
              >>> g = CustomGraph(edges=[[i, i + 1] for i in range(20)])
              >>> hi = CustomHilbert(local_states=[1, -1], graph=g)
              >>> sx_hat = LocalOperator(hi, [sx] * 3, [[0], [1], [5]])
              >>> print(len(sx_hat.acting_on))
              3

              ```
          )EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>,
                    LocalOperator::MatType, LocalOperator::SiteType, double>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("operator"),
           py::arg("acting_on"), py::arg("constant") = 0., R"EOF(
           Constructs a new ``LocalOperator`` given a hilbert space, an
           operator, a site, and (if specified) a constant level
           shift.

           Args:
               hilbert: Hilbert space the operator acts on.
               operator: An operator, in matrix form.
               acting_on: A list of sites, which the corresponding operators act
                   on.
               constant: Level shift for operator. Default is 0.0.

           Examples:
               Constructs a ``LocalOperator`` from a single operator acting on
               a single site.

               ```python
               >>> from netket.graph import CustomGraph
               >>> from netket.hilbert import CustomHilbert
               >>> from netket.operator import LocalOperator
               >>> sx = [[0, 1], [1, 0]]
               >>> g = CustomGraph(edges=[[i, i + 1] for i in range(20)])
               >>> hi = CustomHilbert(local_states=[1, -1], graph=g)
               >>> sx_hat = LocalOperator(hi, sx, [0])
               >>> print(len(sx_hat.acting_on))
               1

               ```
           )EOF")
      .def_property_readonly(
          "local_matrices", &LocalOperator::LocalMatrices,
          R"EOF(list[list]: A list of the local matrices.)EOF")
      .def_property_readonly(
          "acting_on", &LocalOperator::ActingOn,
          R"EOF(list[list]: A list of the sites that each local matrix acts on.)EOF")
      .def("transpose", &LocalOperator::Transpose,
           R"EOF(Returns the transpose of this operator)EOF")
      .def("conjugate", &LocalOperator::Conjugate,
           R"EOF(Returns the complex conjugation of this operator)EOF")
      .def(py::self + py::self)
      .def(
          "__mul__", [](const LocalOperator &a, double b) { return b * a; },
          py::is_operator())
      .def(
          "__rmul__", [](const LocalOperator &a, double b) { return b * a; },
          py::is_operator())
      .def(
          "__mul__", [](const LocalOperator &a, int b) { return b * a; },
          py::is_operator())
      .def(
          "__rmul__", [](const LocalOperator &a, int b) { return b * a; },
          py::is_operator())
      .def(
          "__add__", [](const LocalOperator &a, double b) { return a + b; },
          py::is_operator())
      .def(
          "__add__", [](const LocalOperator &a, int b) { return a + b; },
          py::is_operator())
      .def(
          "__radd__", [](const LocalOperator &a, double b) { return a + b; },
          py::is_operator())
      .def(
          "__radd__", [](const LocalOperator &a, int b) { return a + b; },
          py::is_operator())
      .def(py::self * py::self);
}

}  // namespace netket

#endif
