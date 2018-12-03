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

#define ADDOPERATORMETHODS(name) \
  .def("get_conn", &name::GetConn).def("get_hilbert", &name::GetHilbert)

void AddOperatorModule(py::module &m) {
  auto subm = m.def_submodule("operator");

  py::class_<Operator>(m, "Operator")
      .def(py::init<Ising>())
      .def(py::init<Heisenberg>())
      .def(py::init<BoseHubbard>())
      .def(py::init<GraphHamiltonian>())
      .def(py::init<LocalOperator>()) ADDOPERATORMETHODS(AbstractOperator);

  {
    using DerType = LocalOperator;
    py::class_<DerType>(subm, "LocalOperator")
        .def(py::init<Hilbert, std::vector<DerType::MatType>,
                      std::vector<DerType::SiteType>>(),
             py::arg("hilbert"), py::arg("operators"), py::arg("acting_on"))
        .def(py::init<Hilbert, LocalOperator::MatType,
                      LocalOperator::SiteType>(),
             py::arg("hilbert"), py::arg("operator"), py::arg("acting_on"))
        .def("local_matrices", &LocalOperator::LocalMatrices)
        .def(py::self + py::self)
        .def("__mul__", [](const DerType &a, double b) { return b * a; },
             py::is_operator())
        .def("__rmul__", [](const DerType &a, double b) { return b * a; },
             py::is_operator())
        .def("__mul__", [](const DerType &a, int b) { return b * a; },
             py::is_operator())
        .def("__rmul__", [](const DerType &a, int b) { return b * a; },
             py::is_operator())
        .def(py::self * py::self) ADDOPERATORMETHODS(DerType);
    py::implicitly_convertible<DerType, Operator>();
  }

  {
    using DerType = Ising;
    py::class_<DerType>(subm, "Ising")
        .def(py::init<Hilbert, double, double>(), py::arg("hilbert"),
             py::arg("h"), py::arg("J") = 1.0) ADDOPERATORMETHODS(DerType);
    py::implicitly_convertible<DerType, Operator>();
  }

  {
    using DerType = Heisenberg;
    py::class_<Heisenberg>(subm, "Heisenberg")
        .def(py::init<Hilbert>(), py::arg("hilbert"))
            ADDOPERATORMETHODS(DerType);
    py::implicitly_convertible<DerType, Operator>();
  }

  {
    using DerType = GraphHamiltonian;
    py::class_<DerType>(subm, "GraphHamiltonian")
        .def(py::init<Hilbert, GraphHamiltonian::OVecType,
                      GraphHamiltonian::OVecType, std::vector<int>>(),
             py::arg("hilbert"),
             py::arg("siteops") = GraphHamiltonian::OVecType(),
             py::arg("bondops") = GraphHamiltonian::OVecType(),
             py::arg("bondops_colors") = std::vector<int>())
            ADDOPERATORMETHODS(DerType);
    py::implicitly_convertible<DerType, Operator>();
  }

  {
    using DerType = BoseHubbard;
    py::class_<DerType>(subm, "BoseHubbard")
        .def(py::init<Hilbert, double, double, double>(), py::arg("hilbert"),
             py::arg("U"), py::arg("V") = 0., py::arg("mu") = 0.)
            ADDOPERATORMETHODS(DerType);
    py::implicitly_convertible<DerType, Operator>();
  }

  // Matrix wrappers
  py::class_<AbstractMatrixWrapper<>>(subm, "AbstractMatrixWrapper<>")
      .def("apply", &AbstractMatrixWrapper<>::Apply, py::arg("state"))
      .def_property_readonly("dimension", &AbstractMatrixWrapper<>::Dimension);

  py::class_<SparseMatrixWrapper<>, AbstractMatrixWrapper<>>(
      subm, "SparseMatrixWrapper")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"))
      // property name starts with underscore to mark as internal per PEP8
      .def_property_readonly("_matrix", &SparseMatrixWrapper<>::GetMatrix)
      .def_property_readonly("dimension", &SparseMatrixWrapper<>::Dimension);

  py::class_<DenseMatrixWrapper<>, AbstractMatrixWrapper<>>(
      subm, "DenseMatrixWrapper")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"))
      // property name starts with underscore to mark as internal per PEP8
      .def_property_readonly("_matrix", &DenseMatrixWrapper<>::GetMatrix)
      .def_property_readonly("dimension", &DenseMatrixWrapper<>::Dimension);

  py::class_<DirectMatrixWrapper<>, AbstractMatrixWrapper<>>(
      subm, "DirectMatrixWrapper")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"))
      .def_property_readonly("dimension", &DirectMatrixWrapper<>::Dimension);

  subm.def("wrap_as_matrix", &CreateMatrixWrapper<>, py::arg("operator"),
           py::arg("type") = "Sparse");
}

}  // namespace netket

#endif
