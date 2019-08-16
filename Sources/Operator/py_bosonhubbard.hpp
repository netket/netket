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

#ifndef NETKET_PYBOSONHUBBARD_HPP
#define NETKET_PYBOSONHUBBARD_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "bosonhubbard.hpp"

namespace py = pybind11;

namespace netket {

void AddBoseHubbard(py::module &subm) {
  py::class_<BoseHubbard, AbstractOperator>(
      subm, "BoseHubbard",
      R"EOF(A Bose Hubbard model Hamiltonian operator.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>, double, double,
                    double>(),
           py::keep_alive<1, 2>(), py::arg("hilbert"), py::arg("U"),
           py::arg("V") = 0., py::arg("mu") = 0., R"EOF(
           Constructs a new ``BoseHubbard`` given a hilbert space and a Hubbard
           interaction strength. The chemical potential and the hopping term can
           be specified as well.

           Args:
               hilbert: Hilbert space the operator acts on.
               U: The Hubbard interaction term.
               V: The hopping term.
               mu: The chemical potential.

           Examples:
               Constructs a ``BoseHubbard`` operator for a 2D system.

               ```python
               >>> import netket as nk
               >>> g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
               >>> hi = nk.hilbert.Boson(n_max=3, n_bosons=6, graph=g)
               >>> op = nk.operator.BoseHubbard(U=4.0, hilbert=hi)
               >>> print(op.hilbert.size)
               9

               ```
           )EOF");  // ADDOPERATORMETHODS(BoseHubbard);
}

}  // namespace netket

#endif
