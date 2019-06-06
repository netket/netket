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

#ifndef NETKET_PYSPARSEMATRIXWRAPPER_HPP
#define NETKET_PYSPARSEMATRIXWRAPPER_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "sparse_matrix_wrapper.hpp"

namespace py = pybind11;

namespace netket {

void AddSparseMatrixWrapper(py::module &subm) {
  py::class_<SparseMatrixWrapper<>, AbstractMatrixWrapper<>>(
      subm, "SparseMatrixWrapper",
      R"EOF(This class stores the matrix elements of a given Operator as an Eigen sparse matrix.)EOF")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"),
           R"EOF(
        Constructs a sparse matrix wrapper from an operator. Matrix elements are
        stored as a sparse Eigen matrix.

        Args:
            operator: The operator used to construct the matrix.

        Examples:
            Printing the dimension of a sparse matrix wrapper.

            ```python
            >>> import netket as nk
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> op = nk.operator.Ising(h=1.321, hilbert=hi)
            >>> smw = nk.operator.SparseMatrixWrapper(op)
            >>> print(smw.dimension)
            1048576

            ```
      )EOF")
      .def_property_readonly(
          "data", &SparseMatrixWrapper<>::GetMatrix,
          R"EOF(Eigen SparseMatrix Complex : The stored matrix.)EOF");
}

}  // namespace netket

#endif
