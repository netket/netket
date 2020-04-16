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
namespace py = pybind11;

namespace netket {

void AddOperatorModule(py::module m) {
  auto subm = m.def_submodule("operator");

  subm.def("_rotated_grad_kernel",
           [](Eigen::Ref<const Eigen::ArrayXcd> log_vals_prime,
              Eigen::Ref<const Eigen::ArrayXcd> mels,
              Eigen::Ref<Eigen::ArrayXcd> vec) {
             const auto max_log_val = log_vals_prime.real().maxCoeff();

             vec = (mels * (log_vals_prime - max_log_val).exp()).conjugate();
             vec /= vec.sum();
           });
}
}  // namespace netket

#endif
