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

#ifndef NETKET_PYISING_HPP
#define NETKET_PYISING_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "ising.hpp"

namespace py = pybind11;

namespace netket {

void AddIsing(py::module &subm) {
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
         )EOF");
}

}  // namespace netket

#endif
