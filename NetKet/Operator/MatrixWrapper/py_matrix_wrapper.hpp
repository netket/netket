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

#ifndef NETKET_PYMATRIXWRAPPER_HPP
#define NETKET_PYMATRIXWRAPPER_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "abstract_matrix_wrapper.hpp"
#include "py_dense_matrix_wrapper.hpp"
#include "py_direct_matrix_wrapper.hpp"
#include "py_sparse_matrix_wrapper.hpp"

namespace py = pybind11;

namespace netket {

void AddMatrixWrapper(py::module &subm) {
  py::class_<AbstractMatrixWrapper<>>(subm, "AbstractMatrixWrapper",
                                      R"EOF(This class wraps an AbstractOperator
  and provides a method to apply it to a pure state. @tparam State The type of a
  vector of (complex) coefficients representing the quantum state. Should be
  Eigen::VectorXcd or a compatible type.)EOF")
      .def("apply", &AbstractMatrixWrapper<>::Apply, py::arg("state"))
      .def("dot", &AbstractMatrixWrapper<>::Apply, py::arg("state"))
      .def("matmul", &AbstractMatrixWrapper<>::Apply, py::arg("state"))
      .def_property_readonly(
          "dimension", &AbstractMatrixWrapper<>::Dimension,
          R"EOF(int : The Hilbert space dimension corresponding to the Hamiltonian)EOF");

  AddSparseMatrixWrapper(subm);
  AddDenseMatrixWrapper(subm);
  AddDirectMatrixWrapper(subm);
}

}  // namespace netket

#endif
