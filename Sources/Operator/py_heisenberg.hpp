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

#ifndef NETKET_PYHEISENBERG_HPP
#define NETKET_PYHEISENBERG_HPP

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <complex>
#include <vector>
#include "heisenberg.hpp"

namespace py = pybind11;

namespace netket {

void AddHeisenberg(py::module &subm) {
  py::class_<Heisenberg, AbstractOperator>(
      subm, "Heisenberg", R"EOF(A Heisenberg Hamiltonian operator.)EOF")
      .def(py::init<std::shared_ptr<const AbstractHilbert>>(), py::keep_alive<1, 2>(),
           py::arg("hilbert"), R"EOF(
           Constructs a new ``Heisenberg`` given a hilbert space.

           Args:
               hilbert: Hilbert space the operator acts on.

           Examples:
               Constructs a ``Heisenberg`` operator for a 1D system.

               ```python
               >>> import netket as nk
               >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
               >>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
               >>> op = nk.operator.Heisenberg(hilbert=hi)
               >>> print(op.hilbert.size)
               20

               ```
           )EOF");
}

}  // namespace netket

#endif
