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

#define ADDOPERATORMETHODS(name)              \
  .def("GetConn", &AbstractOperator::GetConn) \
      .def("GetHilbert", &AbstractOperator::GetHilbert)

void AddOperatorModule(py::module &m) {
  auto subm = m.def_submodule("operator");

  py::class_<AbstractOperator, std::shared_ptr<AbstractOperator>>(m, "Operator")
      ADDOPERATORMETHODS(name);

  py::class_<LocalOperator, AbstractOperator, std::shared_ptr<LocalOperator>>(
      subm, "LocalOperator")
      .def(
          py::init<const AbstractHilbert &, std::vector<LocalOperator::MatType>,
                   std::vector<LocalOperator::SiteType>>(),
          py::arg("hilbert"), py::arg("operators"), py::arg("acting_on"))
      .def(py::init<const AbstractHilbert &, LocalOperator::MatType,
                    LocalOperator::SiteType>(),
           py::arg("hilbert"), py::arg("operator"), py::arg("acting_on"))
      .def("GetConn", &LocalOperator::GetConn)
      .def("GetHilbert", &LocalOperator::GetHilbert)
      .def("LocalMatrices", &LocalOperator::LocalMatrices)
      .def(py::self += py::self)
      .def(py::self *= double())
      .def(py::self *= std::complex<double>())
      .def(py::self * py::self) ADDOPERATORMETHODS(LocalOperator);
  // .def(double() * py::self)
  // .def(py::self * double())
  // .def(std::complex<double>() * py::self)
  // .def(py::self * std::complex<double>());

  py::class_<Ising, AbstractOperator, std::shared_ptr<Ising>>(subm, "Ising")
      .def(py::init<const AbstractHilbert &, double, double>(),
           py::arg("hilbert"), py::arg("h"), py::arg("J") = 1.0)
      .def("GetConn", &Ising::GetConn)
      .def("GetHilbert", &Ising::GetHilbert) ADDOPERATORMETHODS(Ising);

  py::class_<Heisenberg, AbstractOperator, std::shared_ptr<Heisenberg>>(
      subm, "Heisenberg")
      .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
      .def("GetConn", &Heisenberg::GetConn)
      .def("GetHilbert", &Heisenberg::GetHilbert)
          ADDOPERATORMETHODS(Heisenberg);

  py::class_<GraphHamiltonian, AbstractOperator,
             std::shared_ptr<GraphHamiltonian>>(subm, "GraphHamiltonian")
      .def(py::init<const AbstractHilbert &, GraphHamiltonian::OVecType,
                    GraphHamiltonian::OVecType, std::vector<int>>(),
           py::arg("hilbert"),
           py::arg("siteops") = GraphHamiltonian::OVecType(),
           py::arg("bondops") = GraphHamiltonian::OVecType(),
           py::arg("bondops_colors") = std::vector<int>())
      .def("GetConn", &GraphHamiltonian::GetConn)
      .def("GetHilbert", &GraphHamiltonian::GetHilbert)
          ADDOPERATORMETHODS(GraphHamiltonian);

  py::class_<BoseHubbard, AbstractOperator, std::shared_ptr<BoseHubbard>>(
      subm, "BoseHubbard")
      .def(py::init<const AbstractHilbert &, double, double, double>(),
           py::arg("hilbert"), py::arg("U"), py::arg("V") = 0.,
           py::arg("mu") = 0.)
      .def("GetConn", &BoseHubbard::GetConn)
      .def("GetHilbert", &BoseHubbard::GetHilbert)
          ADDOPERATORMETHODS(BoseHubbard);

  // Matrix wrappers
  py::class_<AbstractMatrixWrapper<>>(m, "AbstractMatrixWrapper<>")
      .def("apply", &AbstractMatrixWrapper<>::Apply, py::arg("state"))
      .def_property_readonly("dimension", &AbstractMatrixWrapper<>::Dimension);

  py::class_<SparseMatrixWrapper<>, AbstractMatrixWrapper<>>(
      m, "SparseMatrixWrapper")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"))
      // property name starts with underscore to mark as internal per PEP8
      .def_property_readonly("_matrix", &SparseMatrixWrapper<>::GetMatrix)
      .def_property_readonly("dimension", &SparseMatrixWrapper<>::Dimension);

  py::class_<DenseMatrixWrapper<>, AbstractMatrixWrapper<>>(
      m, "DenseMatrixWrapper")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"))
      // property name starts with underscore to mark as internal per PEP8
      .def_property_readonly("_matrix", &DenseMatrixWrapper<>::GetMatrix)
      .def_property_readonly("dimension", &DenseMatrixWrapper<>::Dimension);

  py::class_<DirectMatrixWrapper<>, AbstractMatrixWrapper<>>(
      m, "DirectMatrixWrapper")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"))
      .def_property_readonly("dimension", &DirectMatrixWrapper<>::Dimension);

  m.def("wrap_operator", &CreateMatrixWrapper<>, py::arg("operator"),
        py::arg("type") = "Sparse");
}

}  // namespace netket

#endif
