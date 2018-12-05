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
#include "pylayer.hpp"

namespace py = pybind11;

namespace netket {

#define ADDMACHINEMETHODS(name)                                                \
                                                                               \
  .def_property_readonly("n_par", &name::Npar)                                 \
      .def_property("parameters", &name::GetParameters, &name::SetParameters)  \
      .def("init_random_parameters", &name::InitRandomPars,                    \
           py::arg("seed") = 1234, py::arg("sigma") = 0.1)                     \
      .def("log_val",                                                          \
           (StateType(name::*)(MachineType::VisibleConstType)) & name::LogVal) \
      .def("log_val_diff", (MachineType::VectorType(name::*)(                  \
                               MachineType::VisibleConstType,                  \
                               const std::vector<std::vector<int>> &,          \
                               const std::vector<std::vector<double>> &)) &    \
                               name::LogValDiff)                               \
      .def("der_log",                                                          \
           (MachineType::VectorType(name::*)(MachineType::VisibleConstType)) & \
               name::DerLog)                                                   \
      .def_property_readonly("n_visible", &name::Nvisible)                     \
      .def_property_readonly("hilbert", &name ::GetHilbert)                    \
      .def("save",                                                             \
           [](const name &a, std::string filename) {                           \
             json j;                                                           \
             a.to_json(j);                                                     \
             std::ofstream filewf(filename);                                   \
             filewf << j << std::endl;                                         \
             filewf.close();                                                   \
           })                                                                  \
      .def("load", [](name &a, std::string filename) {                         \
        std::ifstream filewf(filename);                                        \
        if (filewf.is_open()) {                                                \
          json j;                                                              \
          filewf >> j;                                                         \
          filewf.close();                                                      \
          a.from_json(j);                                                      \
        }                                                                      \
      });

void AddMachineModule(py::module &m) {
  auto subm = m.def_submodule("machine");

  py::class_<MachineType>(subm, "Machine")

      ADDMACHINEMETHODS(MachineType);

  {
    using DerMachine = RbmSpin<StateType>;
    py::class_<DerMachine, MachineType>(subm, "RbmSpin")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::keep_alive<1, 2>(), py::arg("hilbert"),
             py::arg("n_hidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_"
                     "bias") = true,
             py::arg("use_hidden_"
                     "bias") = true) ADDMACHINEMETHODS(DerMachine);
  }

  {
    using DerMachine = RbmSpinSymm<StateType>;
    py::class_<DerMachine, MachineType>(subm, "RbmSpinSymm")
        .def(py::init<const AbstractHilbert &, int, bool, bool>(),
             py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("alpha") = 0,
             py::arg("use_visible_"
                     "bias") = true,
             py::arg("use_hidden_"
                     "bias") = true) ADDMACHINEMETHODS(DerMachine);
  }

  {
    using DerMachine = RbmMultival<StateType>;
    py::class_<DerMachine, MachineType>(subm, "RbmMultiVal")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::keep_alive<1, 2>(), py::arg("hilbert"),
             py::arg("n_hidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_"
                     "bias") = true,
             py::arg("use_hidden_"
                     "bias") = true) ADDMACHINEMETHODS(DerMachine);
  }

  {
    using DerMachine = Jastrow<StateType>;
    py::class_<DerMachine, MachineType>(subm, "Jastrow")
        .def(py::init<const AbstractHilbert &>(), py::keep_alive<1, 2>(),
             py::arg("hilbert")) ADDMACHINEMETHODS(DerMachine);
  }

  {
    using DerMachine = JastrowSymm<StateType>;
    py::class_<DerMachine, MachineType>(subm, "JastrowSymm")
        .def(py::init<const AbstractHilbert &>(), py::keep_alive<1, 2>(),
             py::arg("hilbert")) ADDMACHINEMETHODS(DerMachine);
  }

#ifndef COMMA
#define COMMA ,
#endif
  py::class_<MPSPeriodic<StateType, true>, MachineType>(subm,
                                                        "MPSPeriodicDiagonal")
      .def(py::init<const AbstractHilbert &, double, int>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("bond_dim"),
           py::arg("symperiod") = -1)
          ADDMACHINEMETHODS(MPSPeriodic<StateType COMMA true>);

  py::class_<MPSPeriodic<StateType, false>, MachineType>(subm, "MPSPeriodic")
      .def(py::init<const AbstractHilbert &, double, int>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("bond_dim"),
           py::arg("symperiod") = -1)
          ADDMACHINEMETHODS(MPSPeriodic<StateType COMMA false>);

  AddLayerModule(m);

  {
    using DerMachine = FFNN<StateType>;
    py::class_<DerMachine, MachineType>(subm, "FFNN")
        .def(py::init<const AbstractHilbert &,
                      const std::vector<AbstractLayer<StateType> *> &>(),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("hilbert"),
             py::arg("layers")) ADDMACHINEMETHODS(DerMachine);
  }
}

}  // namespace netket

#endif
