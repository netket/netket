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

#ifndef NETKET_PYMACHINE_HPP
#define NETKET_PYMACHINE_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "machine.hpp"
#include "pyactivation.hpp"
#include "pylayer.hpp"

namespace py = pybind11;

namespace netket {

#define ADDMACHINEMETHODS(name)                                             \
                                                                            \
  .def("n_par", &name::Npar)                                                \
      .def("get_parameters", &name::GetParameters)                          \
      .def("set_parameters", &name::SetParameters)                          \
      .def("init_random_parameters", &name::InitRandomPars,                 \
           py::arg("seed") = 1234, py::arg("sigma") = 0.1)                  \
      .def("log_val",                                                       \
           (MachineType(name::*)(AbMachineType::VisibleConstType)) &        \
               name::LogVal)                                                \
      .def("log_val_diff", (AbMachineType::VectorType(name::*)(             \
                               AbMachineType::VisibleConstType,             \
                               const std::vector<std::vector<int>> &,       \
                               const std::vector<std::vector<double>> &)) & \
                               name::LogValDiff)                            \
      .def("der_log", (AbMachineType::VectorType(name::*)(                  \
                          AbMachineType::VisibleConstType)) &               \
                          name::DerLog)                                     \
      .def("n_visible", &name::Nvisible)

void AddMachineModule(py::module &m) {
  auto subm = m.def_submodule("machine");

  py::class_<AbMachineType, std::shared_ptr<AbMachineType>>(subm, "Machine")
      ADDMACHINEMETHODS(AbMachineType);

  {
    using DerMachine = RbmSpin<MachineType>;
    py::class_<DerMachine, AbMachineType, std::shared_ptr<DerMachine>>(
        subm, "RbmSpin")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::arg("hilbert"), py::arg("n_hidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_"
                     "bias") = true,
             py::arg("use_hidden_"
                     "bias") = true) ADDMACHINEMETHODS(DerMachine);
  }

  {
    using DerMachine = RbmSpinSymm<MachineType>;
    py::class_<DerMachine, AbMachineType, std::shared_ptr<DerMachine>>(
        subm, "RbmSpinSymm")
        .def(py::init<const AbstractHilbert &, int, bool, bool>(),
             py::arg("hilbert"), py::arg("alpha") = 0,
             py::arg("use_visible_"
                     "bias") = true,
             py::arg("use_hidden_"
                     "bias") = true) ADDMACHINEMETHODS(DerMachine);
  }

  {
    using DerMachine = RbmMultival<MachineType>;
    py::class_<DerMachine, AbMachineType, std::shared_ptr<DerMachine>>(
        subm, "RbmMultiVal")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::arg("hilbert"), py::arg("n_hidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_"
                     "bias") = true,
             py::arg("use_hidden_"
                     "bias") = true) ADDMACHINEMETHODS(DerMachine);
  }

  {
    using DerMachine = Jastrow<MachineType>;
    py::class_<DerMachine, AbMachineType, std::shared_ptr<DerMachine>>(
        subm, "Jastrow")
        .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
            ADDMACHINEMETHODS(DerMachine);
  }

  {
    using DerMachine = JastrowSymm<MachineType>;
    py::class_<DerMachine, AbMachineType, std::shared_ptr<DerMachine>>(
        subm, "JastrowSymm")
        .def(py::init<const AbstractHilbert &>(), py::arg("hilbert"))
            ADDMACHINEMETHODS(DerMachine);
  }

#ifndef COMMA
#define COMMA ,
#endif
  py::class_<MPSPeriodic<MachineType, true>, AbMachineType,
             std::shared_ptr<MPSPeriodic<MachineType, true>>>(
      subm, "MPSPeriodicDiagonal")
      .def(py::init<const AbstractHilbert &, double, int>(), py::arg("hilbert"),
           py::arg("bond_dim"), py::arg("symperiod") = -1)
          ADDMACHINEMETHODS(MPSPeriodic<MachineType COMMA true>);

  py::class_<MPSPeriodic<MachineType, false>, AbMachineType,
             std::shared_ptr<MPSPeriodic<MachineType, false>>>(subm,
                                                               "MPSPeriodic")
      .def(py::init<const AbstractHilbert &, double, int>(), py::arg("hilbert"),
           py::arg("bond_dim"), py::arg("symperiod") = -1)
          ADDMACHINEMETHODS(MPSPeriodic<MachineType COMMA false>);

  AddActivationModule(m);
  AddLayerModule(m);

  {
    using DerMachine = FFNN<MachineType>;
    py::class_<DerMachine, AbMachineType, std::shared_ptr<DerMachine>>(subm,
                                                                       "FFNN")
        .def(py::init<
                 const AbstractHilbert &,
                 std::vector<std::shared_ptr<AbstractLayer<MachineType>>> &>(),
             py::arg("hilbert"), py::arg("layers"))
            ADDMACHINEMETHODS(DerMachine);
  }
}

}  // namespace netket

#endif
