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

#include "operator.hpp"
#include <complex>
#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

namespace py = pybind11;

namespace netket {

#define ADDOPERATORMETHODS(name)                                               \
  .def("get_conn", &name::GetConn)                                             \
      .def_property_readonly(                                                  \
          "hilbert", &name::GetHilbert,                                        \
          R"EOF(netket.hilbert.Hilbert: ``Hilbert`` space of operator.)EOF")

void AddOperatorModule(py::module &m) {
  auto subm = m.def_submodule("operator");

  py::class_<AbstractOperator>(m, "Operator")
      ADDOPERATORMETHODS(AbstractOperator);

  py::class_<LocalOperator, AbstractOperator>(
      subm, "LocalOperator", R"EOF(A custom local operator.)EOF")
      .def(py::init<const AbstractHilbert &, double>(), py::keep_alive<1, 2>(),
           py::arg("hilbert"), py::arg("constant") = 0., R"EOF(
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
               >>> print(empty_hat.acting_on)
               []

               ```
           )EOF")
      .def(
          py::init<const AbstractHilbert &, std::vector<LocalOperator::MatType>,
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
              >>> print(sx_hat.acting_on)
              [[0], [1], [5]]

              ```
          )EOF")
      .def(py::init<const AbstractHilbert &, LocalOperator::MatType,
                    LocalOperator::SiteType, double>(),
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
               >>> print(sx_hat.acting_on)
               [[0]]

               ```
           )EOF")
      .def_property_readonly(
          "local_matrices", &LocalOperator::LocalMatrices,
          R"EOF(list[list]: A list of the local matrices.)EOF")
      .def_property_readonly(
          "acting_on", &LocalOperator::ActingOn,
          R"EOF(list[list]: A list of the sites that each local matrix acts on.)EOF")
      .def(py::self + py::self)
      .def("__mul__", [](const LocalOperator &a, double b) { return b * a; },
           py::is_operator())
      .def("__rmul__", [](const LocalOperator &a, double b) { return b * a; },
           py::is_operator())
      .def("__mul__", [](const LocalOperator &a, int b) { return b * a; },
           py::is_operator())
      .def("__rmul__", [](const LocalOperator &a, int b) { return b * a; },
           py::is_operator())
      .def("__add__", [](const LocalOperator &a, double b) { return a + b; },
           py::is_operator())
      .def("__add__", [](const LocalOperator &a, int b) { return a + b; },
           py::is_operator())
      .def("__radd__", [](const LocalOperator &a, double b) { return a + b; },
           py::is_operator())
      .def("__radd__", [](const LocalOperator &a, int b) { return a + b; },
           py::is_operator())
      .def(py::self * py::self) ADDOPERATORMETHODS(LocalOperator);

  py::class_<Ising, AbstractOperator>(subm, "Ising",
                                      R"EOF(An Ising Hamiltonian operator.)EOF")
      .def(py::init<const AbstractHilbert &, double, double>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("h"),
           py::arg("J") = 1.0, R"EOF(
           Constructs a new ``Ising`` given a hilbert space, a transverse field,
           and (if specified) a coupling constant.

           Args:
               hilbert: Hilbert space the operator acts on.
               h: The strength of the transverse field.
               J: The strength of the coupling. Default is 1.0.

           Examples:
               Constructs an ``Ising`` operator for a 1D system.

               ```python
               >>> from mpi4py import MPI
               >>> import netket as nk
               >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
               >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
               >>> op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5)
               >>> print(op.hilbert.size)
               20

               ```
           )EOF") ADDOPERATORMETHODS(Ising);

  py::class_<Heisenberg, AbstractOperator>(
      subm, "Heisenberg", R"EOF(A Heisenberg Hamiltonian operator.)EOF")
      .def(py::init<const AbstractHilbert &>(), py::keep_alive<1, 2>(),
           py::arg("hilbert"), R"EOF(
           Constructs a new ``Heisenberg`` given a hilbert space.

           Args:
               hilbert: Hilbert space the operator acts on.

           Examples:
               Constructs a ``Heisenberg`` operator for a 1D system.

               ```python
               >>> from mpi4py import MPI
               >>> import netket as nk
               >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
               >>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
               >>> op = nk.operator.Heisenberg(hilbert=hi)
               >>> print(op.hilbert.size)
               20

               ```
           )EOF") ADDOPERATORMETHODS(Heisenberg);

  py::class_<GraphOperator, AbstractOperator>(
      subm, "GraphOperator", R"EOF(A custom graph based operator.)EOF")
      .def(py::init<const AbstractHilbert &, GraphOperator::OVecType,
                    GraphOperator::OVecType, std::vector<int>>(),
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
               >>> from mpi4py import MPI
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
      .def(py::self + py::self) ADDOPERATORMETHODS(GraphOperator);

  py::class_<BoseHubbard, AbstractOperator>(
      subm, "BoseHubbard",
      R"EOF(A Bose Hubbard model Hamiltonian operator.)EOF")
      .def(py::init<const AbstractHilbert &, double, double, double>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("U"),
           py::arg("V") = 0., py::arg("mu") = 0., R"EOF(
           Constructs a new ``BoseHubbard`` given a hilbert space and a Hubbard
           interaction strength. The chemical potential and the hopping term can
           be specified as well.

           Args:
               hilbert: Hilbert space the operator acts on.
               U: The Hubbard interaction term.
               V: The hopping term.
               mu: The chemical potential.

           Examples:
               Constructs a ``BoseHubbard`` operator for a 2D system.

               ```python
               >>> from mpi4py import MPI
               >>> import netket as nk
               >>> g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
               >>> hi = nk.hilbert.Boson(n_max=3, n_bosons=6, graph=g)
               >>> op = nk.operator.BoseHubbard(U=4.0, hilbert=hi)
               >>> print(op.hilbert.size)
               9

               ```
           )EOF") ADDOPERATORMETHODS(BoseHubbard);

// Matrix wrappers
#define ADDWRAPPERMETHODS(name)                                                \
  .def_property_readonly(                                                      \
      "dimension", &name<>::Dimension,                                         \
      R"EOF(int : The Hilbert space dimension corresponding to the Hamiltonian)EOF")

  py::class_<AbstractMatrixWrapper<>>(subm, "AbstractMatrixWrapper<>",
                                      R"EOF(This class wraps an AbstractOperator
  and provides a method to apply it to a pure state. @tparam State The type of a
  vector of (complex) coefficients representing the quantum state. Should be
  Eigen::VectorXcd or a compatible type.)EOF")
      .def("apply", &AbstractMatrixWrapper<>::Apply, py::arg("state"))
          ADDWRAPPERMETHODS(AbstractMatrixWrapper);

  py::class_<SparseMatrixWrapper<>, AbstractMatrixWrapper<>>(
      subm, "SparseMatrixWrapper",
      R"EOF(This class stores the matrix elements of a given Operator as an Eigen sparse matrix.)EOF")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"), R"EOF(
        Constructs a sparse matrix wrapper from an operator. Matrix elements are
        stored as a sparse Eigen matrix.

        Args:
            operator: The operator used to construct the matrix.

        Examples:
            Printing the dimension of a sparse matrix wrapper.

            ```python
            >>> import netket as nk
            >>> from mpi4py import MPI
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> op = nk.operator.Ising(h=1.321, hilbert=hi)
            >>> smw = nk.operator.SparseMatrixWrapper(op)
            >>> smw.dimension
            1048576

            ```
      )EOF")
      // property name starts with underscore to mark as internal per PEP8
      .def_property_readonly(
          "_matrix", &SparseMatrixWrapper<>::GetMatrix,
          R"EOF(Eigen SparseMatrix Complex : The stored matrix.)EOF")
          ADDWRAPPERMETHODS(SparseMatrixWrapper);

  py::class_<DenseMatrixWrapper<>, AbstractMatrixWrapper<>>(
      subm, "DenseMatrixWrapper",
      R"EOF(This class stores the matrix elements of
        a given Operator as an Eigen dense matrix.)EOF")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"), R"EOF(
        Constructs a dense matrix wrapper from an operator. Matrix elements are
        stored as a dense Eigen matrix.

        Args:
            operator: The operator used to construct the matrix.

        Examples:
            Printing the dimension of a dense matrix wrapper.

            ```python
            >>> import netket as nk
            >>> from mpi4py import MPI
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> op = nk.operator.Ising(h=1.321, hilbert=hi)
            >>> dmw = nk.operator.DirectMatrixWrapper(op)
            >>> dmw.dimension
            1048576

            ```

      )EOF")
      // property name starts with underscore to mark as internal per PEP8
      .def_property_readonly("_matrix", &DenseMatrixWrapper<>::GetMatrix,
                             R"EOF(Eigen MatrixXcd : The stored matrix.)EOF")
          ADDWRAPPERMETHODS(DenseMatrixWrapper);

  py::class_<DirectMatrixWrapper<>, AbstractMatrixWrapper<>>(
      subm, "DirectMatrixWrapper",
      R"EOF(This class wraps a given Operator. The
        matrix elements are not stored separately but are computed from
        Operator::FindConn every time Apply is called.)EOF")
      .def(py::init<const AbstractOperator &>(), py::arg("operator"), R"EOF(
        Constructs a direct matrix wrapper from an operator. Matrix elements are
        calculated when required.

        Args:
            operator: The operator used to construct the matrix.

        Examples:
            Printing the dimension of a direct matrix wrapper.

            ```python
            >>> import netket as nk
            >>> from mpi4py import MPI
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> op = nk.operator.Ising(h=1.321, hilbert=hi)
            >>> dmw = nk.operator.DirectMatrixWrapper(op)
            >>> dmw.dimension
            1048576

            ```

      )EOF") ADDWRAPPERMETHODS(DirectMatrixWrapper);

  subm.def("wrap_as_matrix", &CreateMatrixWrapper<>, py::arg("operator"),
           py::arg("type") = "Sparse");
}

} // namespace netket

#endif
