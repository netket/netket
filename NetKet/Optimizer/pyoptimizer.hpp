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

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "optimizer.hpp"

namespace py = pybind11;

namespace netket {

#define ADDOPTIMIZERMETHODS(name) .def("reset", &name::Reset);

void AddOptimizerModule(py::module &m) {
  auto subm = m.def_submodule("optimizer");

  py::class_<AbstractOptimizer>(subm, "Optimizer")
      ADDOPTIMIZERMETHODS(AbstractOptimizer);

  {
    using OptType = Sgd;
    py::class_<OptType, AbstractOptimizer>(subm, "Sgd")
        .def(py::init<double, double, double>(), py::arg("learning_rate"),
             py::arg("l2_reg") = 0, py::arg("decay_factor") = 1.0)
            ADDOPTIMIZERMETHODS(OptType);
  }

  {
    using OptType = RMSProp;
    py::class_<OptType, AbstractOptimizer>(subm, "RMSProp")
        .def(py::init<double, double, double>(),
             py::arg("learning_rate") = 0.001, py::arg("beta") = 0.9,
             py::arg("epscut") = 1.0e-7) ADDOPTIMIZERMETHODS(OptType);
  }
  {
    using OptType = Momentum;
    py::class_<OptType, AbstractOptimizer>(subm, "Momentum")
        .def(py::init<double, double>(), py::arg("learning_rate") = 0.001,
             py::arg("beta") = 0.9) ADDOPTIMIZERMETHODS(OptType);
  }
  {
    using OptType = AMSGrad;
    py::class_<OptType, AbstractOptimizer>(subm, "AMSGrad")
        .def(py::init<double, double, double, double>(),
             py::arg("learning_rate") = 0.001, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("epscut") = 1.0e-7)
            ADDOPTIMIZERMETHODS(OptType);
  }
  {
    using OptType = AdaMax;
    py::class_<OptType, AbstractOptimizer>(subm, "AdaMax")
        .def(py::init<double, double, double, double>(),
             py::arg("alpha") = 0.001, py::arg("beta1") = 0.9,
             py::arg("beta2") = 0.999, py::arg("epscut") = 1.0e-7)
            ADDOPTIMIZERMETHODS(OptType);
  }
  {
    using OptType = AdaGrad;
    py::class_<OptType, AbstractOptimizer>(subm, "AdaGrad")
        .def(py::init<double, double>(), py::arg("learning_rate") = 0.001,
             py::arg("epscut") = 1.0e-7) ADDOPTIMIZERMETHODS(OptType);
  }
  {
    using OptType = AdaDelta;
    py::class_<OptType, AbstractOptimizer>(subm, "AdaDelta")
        .def(py::init<double, double>(), py::arg("rho") = 0.95,
             py::arg("epscut") = 1.0e-7) ADDOPTIMIZERMETHODS(OptType);
  }
}

}  // namespace netket

#endif
