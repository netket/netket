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

#ifndef NETKET_PYJASTROW_HPP
#define NETKET_PYJASTROW_HPP

#include <mpi.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "jastrow.hpp"

namespace py = pybind11;

namespace netket {

void AddJastrow(py::module &subm) {
  py::class_<Jastrow<StateType>, MachineType>(subm, "Jastrow", R"EOF(
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
                     >>> print(ma.n_par)
                     190

                     ```
                 )EOF");
}

}  // namespace netket

#endif
