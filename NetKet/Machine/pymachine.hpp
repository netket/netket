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

void AddMachineModule(py::module &m) {
  auto subm = m.def_submodule("machine");

  py::class_<MachineType>(subm, "Machine")
      .def_property_readonly(
          "n_par", &MachineType::Npar,
          R"EOF(int: The number of parameters in the machine.)EOF")
      .def_property("parameters", &MachineType::GetParameters,
                    &MachineType::SetParameters,
                    R"EOF(list: List containing the parameters within the layer.
            Read and write)EOF")
      .def("init_random_parameters", &MachineType::InitRandomPars,
           py::arg("seed") = 1234, py::arg("sigma") = 0.1,
           R"EOF(
             Member function to initialise machine parameters.

             Args:
                 seed: The random number generator seed.
                 sigma: Standard deviation of normal distribution from which
                     parameters are drawn.
           )EOF")
      .def("log_val",
           (StateType(MachineType::*)(MachineType::VisibleConstType)) &
               MachineType::LogVal,
           py::arg("v"),
           R"EOF(
                 Member function to obtain log value of machine given an input
                 vector.

                 Args:
                     v: Input vector to machine.
           )EOF")
      .def("log_val_diff",
           (MachineType::VectorType(MachineType::*)(
               MachineType::VisibleConstType,
               const std::vector<std::vector<int>> &,
               const std::vector<std::vector<double>> &)) &
               MachineType::LogValDiff,
           py::arg("v"), py::arg("tochange"), py::arg("newconf"),
           R"EOF(
                 Member function to obtain difference in log value of machine
                 given an input and a change to the input.

                 Args:
                     v: Input vector to machine.
                     tochange: list containing the indices of the input to be
                         changed
                     newconf: list containing the new (changed) values at the
                         indices specified in tochange
           )EOF")
      .def("der_log",
           (MachineType::VectorType(MachineType::*)(
               MachineType::VisibleConstType)) &
               MachineType::DerLog,
           py::arg("v"),
           R"EOF(
                 Member function to obtain the derivatives of log value of
                 machine given an input wrt the machine's parameters.

                 Args:
                     v: Input vector to machine.
           )EOF")
      .def_property_readonly(
          "n_visible", &MachineType::Nvisible,
          R"EOF(int: The number of inputs into the machine aka visible units in
            the case of Restricted Boltzmann Machines.)EOF")
      .def_property_readonly("hilbert", &MachineType ::GetHilbert,
                             R"EOF(The hilbert space object of the system.)EOF")
      .def("save",
           [](const MachineType &a, std::string filename) {
             json j;
             a.to_json(j);
             std::ofstream filewf(filename);
             filewf << j << std::endl;
             filewf.close();
           },
           py::arg("filename"),
           R"EOF(
                 Member function to save the machine parameters.

                 Args:
                     filename: name of file to save parameters to.
           )EOF")
      .def("load",
           [](MachineType &a, std::string filename) {
             std::ifstream filewf(filename);
             if (filewf.is_open()) {
               json j;
               filewf >> j;
               filewf.close();
               a.from_json(j);
             }
           },
           py::arg("filename"),
           R"EOF(
                 Member function to load machine parameters from a json file.

                 Args:
                     filename: name of file to load parameters from.
           )EOF");

  {
    using DerMachine = RbmSpin<StateType>;
    py::class_<DerMachine, MachineType>(subm, "RbmSpin", R"EOF(
          A fully connected Restricted Boltzmann Machine (RBM). This type of
          RBM has spin 1/2 hidden units and is defined by:

          $$ \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M \cosh
             \left(\sum_i^N W_{ij} s_i + b_j \right) $$

          for arbitrary local quantum numbers $$ s_i $$.)EOF")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::keep_alive<1, 2>(), py::arg("hilbert"),
             py::arg("n_hidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_bias") = true,
             py::arg("use_hidden_bias") = true,
             R"EOF(
                   Constructs a new ``RbmSpin`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.
                       n_hidden: Number of hidden units.
                       alpha: Hidden unit density.
                       use_visible_bias: If ``True`` then there would be a
                                        bias on the visible units.
                                        Default ``True``.
                       use_hidden_bias: If ``True`` then there would be a
                                       bias on the visible units.
                                       Default ``True``.

                   Examples:
                       A ``RbmSpin`` machine with hidden unit density
                       alpha = 2 for a one-dimensional L=20 spin-half system:

                       ```python
                       >>> from netket.machine import RbmSpin
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = RbmSpin(alpha=2)
                       ```
                   )EOF");
  }

  {
    using DerMachine = RbmSpinSymm<StateType>;
    py::class_<DerMachine, MachineType>(subm, "RbmSpinSymm", R"EOF(
             A fully connected Restricted Boltzmann Machine with lattice
             symmetries. This type of RBM has spin 1/2 hidden units and is
             defined by:

             $$ \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M
                \cosh \left(\sum_i^N W_{ij} s_i + b_j \right) $$

             for arbitrary local quantum numbers $$ s_i $$. However, the weights
             ($$ W_{ij} $$) and biases ($$ a_i $$, $$ b_i $$) respects the
             specified symmetries of the lattice.)EOF")
        .def(py::init<const AbstractHilbert &, int, bool, bool>(),
             py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("alpha") = 0,
             py::arg("use_visible_bias") = true,
             py::arg("use_hidden_bias") = true,
             R"EOF(
                   Constructs a new ``RbmSpinSymm`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.
                       alpha: Hidden unit density.
                       use_visible_bias: If ``True`` then there would be a
                                        bias on the visible units.
                                        Default ``True``.
                       use_hidden_bias: If ``True`` then there would be a
                                       bias on the visible units.
                                       Default ``True``.

                   Examples:
                       A ``RbmSpinSymm`` machine with hidden unit density
                       alpha = 2 for a one-dimensional L=20 spin-half system:

                       ```python
                       >>> from netket.machine import RbmSpinSymm
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = RbmSpinSymm(hilbert=hi, alpha=2)
                       ```
                   )EOF");
  }

  {
    using DerMachine = RbmMultival<StateType>;
    py::class_<DerMachine, MachineType>(subm, "RbmMultiVal", R"EOF(
             A fully connected Restricted Boltzmann Machine for handling larger
             local Hilbert spaces.)EOF")
        .def(py::init<const AbstractHilbert &, int, int, bool, bool>(),
             py::keep_alive<1, 2>(), py::arg("hilbert"),
             py::arg("n_hidden") = 0, py::arg("alpha") = 0,
             py::arg("use_visible_"
                     "bias") = true,
             py::arg("use_hidden_"
                     "bias") = true);
  }

  {
    using DerMachine = Jastrow<StateType>;
    py::class_<DerMachine, MachineType>(subm, "Jastrow", R"EOF(
             A Jastrow wavefunction Machine. This machine defines the following
             wavefunction:

             $$ \Psi(s_1,\dots s_N) = e^{\sum_{ij} s_i W_{ij} s_i}$$

             where $$ W_{ij} $$ are the Jastrow parameters.
             )EOF")
        .def(py::init<const AbstractHilbert &>(), py::keep_alive<1, 2>(),
             py::arg("hilbert"), R"EOF(
                   Constructs a new ``Jastrow`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.

                   Examples:
                       A ``Jastrow`` machine for a one-dimensional L=20 spin 1/2
                       system:

                       ```python
                       >>> from netket.machine import Jastrow
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = Jastrow(hilbert=hi)
                       ```
                   )EOF");
  }

  {
    using DerMachine = JastrowSymm<StateType>;
    py::class_<DerMachine, MachineType>(subm, "JastrowSymm", R"EOF(
             A Jastrow wavefunction Machine with lattice symmetries.This machine
             defines the wavefunction as follows:

             $$ \Psi(s_1,\dots s_N) = e^{\sum_{ij} s_i W_{ij} s_i}$$

             where $$ W_{ij} $$ are the Jastrow parameters respects the
             specified symmetries of the lattice.)EOF")
        .def(py::init<const AbstractHilbert &>(), py::keep_alive<1, 2>(),
             py::arg("hilbert"), R"EOF(
                   Constructs a new ``JastrowSymm`` machine:

                   Args:
                       hilbert: Hilbert space object for the system.

                   Examples:
                       A ``JastrowSymm`` machine for a one-dimensional L=20 spin
                       1/2 system:

                       ```python
                       >>> from netket.machine import JastrowSymm
                       >>> from netket.hilbert import Spin
                       >>> from netket.graph import Hypercube
                       >>> g = Hypercube(length=20, n_dim=1)
                       >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                       >>> ma = JastrowSymm(hilbert=hi)
                       ```
                   )EOF");
  }

#ifndef COMMA
#define COMMA ,
#endif
  py::class_<MPSPeriodic<StateType, true>, MachineType>(subm,
                                                        "MPSPeriodicDiagonal")
      .def(py::init<const AbstractHilbert &, double, int>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("bond_dim"),
           py::arg("symperiod") = -1);

  py::class_<MPSPeriodic<StateType, false>, MachineType>(subm, "MPSPeriodic")
      .def(py::init<const AbstractHilbert &, double, int>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("bond_dim"),
           py::arg("symperiod") = -1);

  AddLayerModule(m);

  {
    using DerMachine = FFNN<StateType>;
    py::class_<DerMachine, MachineType>(subm, "FFNN", R"EOF(
             A feedforward neural network (FFNN) Machine. This machine is
             constructed by providing a sequence of layers from the ``layer``
             class. Each layer implements a transformation such that the
             information is transformed sequentially as it moves from the input
             nodes through the hidden layers and to the output nodes.)EOF")
        .def(py::init([](AbstractHilbert const &hi, py::tuple tuple) {
               auto layers =
                   py::cast<std::vector<AbstractLayer<StateType> *>>(tuple);
               return DerMachine{hi, std::move(layers)};
             }),
             py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::arg("hilbert"),
             py::arg("layers"),
             R"EOF(
                    Constructs a new ``FFNN`` machine:

                    Args:
                        hi: Hilbert space object for the system.
                        layers: Tuple of layers.

                    Examples:
                        A ``FFNN`` machine with 2 layers.
                        for a one-dimensional L=20 spin-half system:

                        ```python
                        >>> from netket.layer import SumOutput
                        >>> from netket.layer import FullyConnected
                        >>> from netket.layer import Lncosh
                        >>> from netket.hilbert import Spin
                        >>> from netket.graph import Hypercube
                        >>> from netket.machine import FFNN
                        >>> g = Hypercube(length=20, n_dim=1)
                        >>> hi = Spin(s=0.5, total_sz=0, graph=g)
                        >>> layers = (
                          FullyConnected(input_size=20,output_size=20,use_bias=True),
                          Lncosh(input_size=20),
                          SumOutput(input_size=20))
                        >>> ma = FFNN(hi, layers)
                        ```
                    )EOF");
  }
}

}  // namespace netket

#endif
