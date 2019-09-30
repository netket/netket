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

#ifndef NETKET_PYOPTIMIZER_HPP
#define NETKET_PYOPTIMIZER_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "abstract_optimizer.hpp"
#include "py_ada_delta.hpp"
#include "py_ada_grad.hpp"
#include "py_ada_max.hpp"
#include "py_ams_grad.hpp"
#include "py_momentum.hpp"
#include "py_rms_prop.hpp"
#include "py_sgd.hpp"
#include "py_stochastic_reconfiguration.hpp"

namespace py = pybind11;

namespace netket {

void AddOptimizerModule(py::module &m) {
  auto subm = m.def_submodule("optimizer");

  using UpdateReal =
      void (AbstractOptimizer::*)(const VectorXd &, Eigen::Ref<VectorXd>);
  using UpdateCplx =
      void (AbstractOptimizer::*)(const VectorXcd &, Eigen::Ref<VectorXcd>);

  py::class_<AbstractOptimizer>(subm, "Optimizer")
      .def("reset", &AbstractOptimizer::Reset, R"EOF(
       Member function resetting the internal state of the optimizer.)EOF")
      .def("update", static_cast<UpdateReal>(&AbstractOptimizer::Update),
           py::arg("grad").noconvert(), py::arg("param").noconvert(),
           "Update `param` by applying a gradient-based optimization step "
           "using `grad`.")
      .def("update", static_cast<UpdateCplx>(&AbstractOptimizer::Update),
           py::arg("grad").noconvert(), py::arg("param").noconvert(),
           "Update `param` by applying a gradient-based optimization step "
           "using `grad`.");

  AddSgd(subm);
  AddRmsProp(subm);
  AddMomentum(subm);
  AddAmsGrad(subm);
  AddAdaMax(subm);
  AddAdaGrad(subm);
  AddAdaDelta(subm);
  AddSR(subm);
}

}  // namespace netket

#endif
