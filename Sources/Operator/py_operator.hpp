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
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <complex>
#include <tuple>
#include <vector>
#include "abstract_operator.hpp"
#include "py_bosonhubbard.hpp"
#include "py_graph_operator.hpp"
#include "py_local_liouvillian.hpp"
#include "py_local_operator.hpp"
#include "py_pauli_strings.hpp"
namespace py = pybind11;

namespace netket {

void AddOperatorModule(py::module m) {
  auto subm = m.def_submodule("operator");

  auto op =
      py::class_<AbstractOperator, std::shared_ptr<AbstractOperator>>(
          m, "Operator", R"EOF(
      Abstract class for quantum Operators. This class prototypes the methods
      needed by a class satisfying the Operator concept. Users interested in
      implementing new quantum Operators should derive they own class from this
      class
       )EOF")
          .def("get_conn",
               [](AbstractOperator& op,
                  py::array_t<double, py::array::c_style> samples) {
                 switch (samples.ndim()) {
                   case 2:
                     return py::cast(
                         op.GetConn(Eigen::Map<const RowMatrix<double>>{
                             samples.data(), samples.shape(0),
                             samples.shape(1)}));
                   case 1: {
                     auto conns =
                         op.GetConn(Eigen::Map<const RowMatrix<double>>{
                             samples.data(), 1, samples.shape(0)});

                     return py::cast(
                         std::tuple<RowMatrix<double>, Eigen::VectorXcd>(
                             std::get<0>(conns)[0], std::get<1>(conns)[0]));
                   }
                   default:
                     NETKET_CHECK(false, InvalidInputError,
                                  "samples has wrong dimension: "
                                      << samples.ndim()
                                      << "; expected either 1 or 2");
                 }
               },
               py::arg("v"), R"EOF(
       Member function finding the connected elements of the Operator. Starting
       from a given visible state v, it finds all other visible states v' such
       that the matrix element :math:`O(v,v')` is different from zero. In general there
       will be several different connected visible units satisfying this
       condition, and they are denoted here :math:`v'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

       Args:
           v: A constant reference to the visible configuration.

       )EOF")
          .def("get_conn_flattened", &AbstractOperator::GetConnFlattened,
               py::arg("v"), py::arg("sections"))
          .def("get_n_conn", &AbstractOperator::GetNConn, py::arg("v"),
               py::arg("n_conn"))
          .def_property_readonly(
              "hilbert", &AbstractOperator::GetHilbert,
              R"EOF(netket.hilbert.Hilbert: ``Hilbert`` space of operator.)EOF")
          .def("to_sparse", &AbstractOperator::ToSparse,
               R"EOF(
         Returns the sparse matrix representation of the operator. Note that, in general,
         the size of the matrix is exponential in the number of quantum
         numbers, and this operation should thus only be performed for
         low-dimensional Hilbert spaces or sufficiently sparse operators.

         This method requires an indexable Hilbert space.
         )EOF")
          .def("to_dense", &AbstractOperator::ToDense,
               R"EOF(
         Returns the dense matrix representation of the operator. Note that, in general,
         the size of the matrix is exponential in the number of quantum
         numbers, and this operation should thus only be performed for
         low-dimensional Hilbert spaces.

         This method requires an indexable Hilbert space.
         )EOF")
          .def("to_linear_operator",
               [](py::object py_self) {
                 const auto* cxx_self = py_self.cast<AbstractOperator const*>();
                 const auto dtype =
                     py::module::import("numpy").attr("complex128");
                 const auto linear_operator =
                     py::module::import("scipy.sparse.linalg")
                         .attr("LinearOperator");
                 const auto dim = cxx_self->Dimension();
                 return linear_operator(
                     py::arg{"shape"} = std::make_tuple(dim, dim),
                     py::arg{"matvec"} = py::cpp_function(
                         // TODO: Does this copy data?
                         [py_self, cxx_self](const Eigen::VectorXcd& x) {
                           return cxx_self->Apply(x);
                         }),
                     py::arg{"dtype"} = dtype);
               },
               R"EOF(
        Converts `Operator` to `scipy.sparse.linalg.LinearOperator`.

        This method requires an indexable Hilbert space.
          )EOF")
          .def("__call__", &AbstractOperator::Apply,
               R"EOF(
        Applies the operator to a state.
            )EOF");

  AddBoseHubbard(subm);
  AddLocalOperator(subm);
  AddGraphOperator(subm);
  AddPauliStrings(subm);
  AddLocalSuperOperatorModule(subm);

  subm.def("_rotated_grad_kernel",
           [](Eigen::Ref<const Eigen::ArrayXcd> log_vals_prime,
              Eigen::Ref<const Eigen::ArrayXcd> mels,
              Eigen::Ref<Eigen::ArrayXcd> vec) {
             const auto max_log_val = log_vals_prime.real().maxCoeff();

             vec = (mels * (log_vals_prime - max_log_val).exp()).conjugate();
             vec /= vec.sum();
           });

  subm.def(
      "_der_local_values_notcentered_kernel",
      [](Eigen::Ref<const Eigen::VectorXcd> log_vals_zero,
         const std::vector<Eigen::Ref<const Eigen::VectorXcd>>& log_vals_prime,
         const std::vector<Eigen::Ref<const Eigen::VectorXcd>>& mels,
         const std::vector<Eigen::Ref<const RowMatrix<Complex>>>& der_log_vals,
         //   Eigen::Ref<Eigen::VectorXcd> local_vals,
         Eigen::Ref<RowMatrix<Complex>> der_local_vals) {
        Eigen::VectorXcd tmp = Eigen::VectorXcd(1);
        for (std::size_t k = 0; k < mels.size(); k++) {
          tmp.resize(log_vals_prime[k].size());

          tmp = (mels[k].array() *
                 (log_vals_prime[k].array() - log_vals_zero(k)).exp());

          // Computing the local_val is not needed, but it's almost for free so
          // we might as well return this too.
          // local_vals(k) = tmp.sum();

          der_local_vals.row(k) =
              (der_log_vals[k].array().colwise() * tmp.array()).colwise().sum();
        }
      });

  subm.def(
      "_der_local_values_kernel",
      [](Eigen::Ref<const Eigen::VectorXcd> log_vals_zero,
         const std::vector<Eigen::Ref<const Eigen::VectorXcd>>& log_vals_prime,
         const std::vector<Eigen::Ref<const Eigen::VectorXcd>>& mels,
         const Eigen::Ref<const RowMatrix<Complex>>& der_log_zero,
         const std::vector<Eigen::Ref<const RowMatrix<Complex>>>& der_log_vals,
         //   Eigen::Ref<Eigen::VectorXcd> local_vals,
         Eigen::Ref<RowMatrix<Complex>> der_local_vals) {
        for (std::size_t k = 0; k < mels.size(); k++) {
          auto tmp = (mels[k].array() *
                      (log_vals_prime[k].array() - log_vals_zero(k)).exp());

          // Computing the local_val is not needed, but it's almost for free so
          // we might as well return this too.
          // local_vals(k) = tmp.sum();

          der_local_vals.row(k) =
              ((der_log_vals[k].colwise() - der_log_zero.col(k))
                   .array()
                   .colwise() *
               tmp.array())
                  .colwise()
                  .sum();
        }
      });
}
}  // namespace netket

#endif
