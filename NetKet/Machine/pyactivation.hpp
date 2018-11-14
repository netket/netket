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

#ifndef NETKET_PYACTIVATION_HPP
#define NETKET_PYACTIVATION_HPP

#include <mpi.h>
#include "activations.hpp"

namespace py = pybind11;

namespace netket {

#define ADDACTIVATIONMETHODS(name)    \
                                      \
  .def("__call__", &name::operator()) \
      .def("ApplyJacobian", &name::ApplyJacobian);

void AddActivationModule(py::module &m) {
  auto subm = m.def_submodule("activation");

  py::class_<AbstractActivation>(subm, "Activation")
      ADDACTIVATIONMETHODS(AbstractActivation);
  py::class_<Tanh, AbstractActivation>(subm, "Tanh")
      .def(py::init<>()) ADDACTIVATIONMETHODS(Tanh);
  py::class_<Lncosh, AbstractActivation>(subm, "Lncosh")
      .def(py::init<>()) ADDACTIVATIONMETHODS(Lncosh);
  py::class_<Identity, AbstractActivation>(subm, "Identity")
      .def(py::init<>()) ADDACTIVATIONMETHODS(Identity);
}

}  // namespace netket

#endif
