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

#ifndef NETKET_PYOPERATOR_HPP
#define NETKET_PYOPERATOR_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "MatrixWrapper/py_matrix_wrapper.hpp"
#include "abstract_operator.hpp"
#include "py_bosonhubbard.hpp"
#include "py_graph_operator.hpp"
#include "py_local_operator.hpp"

namespace py = pybind11;

namespace netket {

void AddOperatorModule(py::module &m) {
  auto subm = m.def_submodule("operator");

  auto op =
      py::class_<AbstractOperator>(m, "Operator", R"EOF(
      Abstract class for quantum Operators. This class prototypes the methods
      needed by a class satisfying the Operator concept. Users interested in
      implementing new quantum Operators should derive they own class from this
      class
       )EOF")
          .def("get_conn", &AbstractOperator::GetConn, py::arg("v"), R"EOF(
       Member function finding the connected elements of the Operator. Starting
       from a given visible state v, it finds all other visible states v' such
       that the matrix element O(v,v') is different from zero. In general there
       will be several different connected visible units satisfying this
       condition, and they are denoted here v'(k), for k=0,1...N_connected.

       Args:
           v: A constant reference to the visible configuration.

       )EOF")
          .def_property_readonly(
              "hilbert", &AbstractOperator::GetHilbert,
              R"EOF(netket.hilbert.Hilbert: ``Hilbert`` space of operator.)EOF")
          .def(
              "to_sparse",
              [](const AbstractOperator &self) {
                return SparseMatrixWrapper<>(self).GetMatrix();
              },
              R"EOF(
         Returns the sparse matrix representation of the operator. Note that, in general,
         the size of the matrix is exponential in the number of quantum
         numbers, and this operation should thus only be performed for
         low-dimensional Hilbert spaces or sufficiently sparse operators.

         This method requires an indexable Hilbert space.
         )EOF")
          .def(
              "to_dense",
              [](const AbstractOperator &self) {
                return DenseMatrixWrapper<>(self).GetMatrix();
              },
              R"EOF(
         Returns the dense matrix representation of the operator. Note that, in general,
         the size of the matrix is exponential in the number of quantum
         numbers, and this operation should thus only be performed for
         low-dimensional Hilbert spaces.

         This method requires an indexable Hilbert space.
         )EOF");

  AddBoseHubbard(subm);
  AddLocalOperator(subm);
  AddGraphOperator(subm);
  AddMatrixWrapper(subm);
}

}  // namespace netket

#endif
